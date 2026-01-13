from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import time
import torch
import torch.nn.functional as F

from .rollout import RolloutBatch, ForwardStats

@dataclass
class PPOStats:
    loss_total: float
    loss_ppo: float
    loss_kl: float
    approx_kl_new_old: float
    approx_kl_new_ref: float
    k2_ref_old: float
    entropy: float
    grad_norm: float
    step_time_s: float
    train_policy_forwards: int
    train_ref_forwards: int
    train_policy_forward_time_s: float
    train_ref_forward_time_s: float

def _teacher_logp(model, input_ids: torch.Tensor, stats: ForwardStats, which: str) -> torch.Tensor:
    """Return log-prob for each token position under model, aligned to positions of tokens.

    logp[:, t] is log prob of input_ids[:, t] under logits at t-1. logp[:, 0] = 0.
    """
    t0 = time.perf_counter()
    logits, _ = model(input_ids, None)
    dt = time.perf_counter() - t0
    if which == "policy":
        stats.policy_forwards += 1
        stats.policy_forward_time_s += dt
    else:
        stats.ref_forwards += 1
        stats.ref_forward_time_s += dt
    logprobs = F.log_softmax(logits, dim=-1)
    B, T = input_ids.shape
    out = torch.zeros((B, T), dtype=torch.float32, device=input_ids.device)
    if T > 1:
        nxt = input_ids[:, 1:]
        lp = logprobs[:, :-1, :].gather(2, nxt[:, :, None]).squeeze(2)
        out[:, 1:] = lp
    return out

def ppo_update(
    policy,
    ref_policy,
    batch: RolloutBatch,
    optimizer,
    *,
    ppo_epochs: int,
    minibatch_size: int,
    clip_eps_low: float,
    clip_eps_high: float,
    beta_kl: float,
    variant: str,
    ref_compute: str,
    kl_estimator: str,
    k2_mode: str,
    max_grad_norm: float,
) -> PPOStats:
    """One PPO update over a rollout batch.

    variant:
      - 'A': explicit reference KL regularization
      - 'B': no reference model; regularize vs old policy only
    ref_compute:
      - 'cached': use batch.logp_ref
      - 'naive_epoch': recompute ref logp inside each PPO epoch (intentionally wasteful)
    kl_estimator:
      - 'simple': tokenwise (logp_new - logp_ref) or (logp_new - logp_old) if B
      - 'k2': compute k2 estimate (always logged); penalty uses k2 iff k2_mode=='theta_over_ref'
    k2_mode:
      - 'ref_over_old': x = exp(clamp(logp_ref - logp_old))  (as written in NVIDIA ProRL v2 blog)
      - 'theta_over_ref': x = exp(clamp(logp_new - logp_ref)) (differentiable)
    """
    t_start = time.perf_counter()
    device = batch.input_ids.device
    B, T = batch.input_ids.shape
    variant = variant.upper().strip()
    ref_compute = ref_compute.lower().strip()
    kl_estimator = kl_estimator.lower().strip()
    k2_mode = k2_mode.lower().strip()

    # advantage: REINFORCE++-lite: batch normalize rewards
    rewards = batch.rewards  # [B]
    adv = rewards - rewards.mean()
    adv = adv / (rewards.std(unbiased=False) + 1e-8)
    adv_tok = adv[:, None].expand(B, T)

    action_mask = batch.action_mask
    idx = action_mask.nonzero(as_tuple=False)  # [N,2]
    N = idx.shape[0]
    if N == 0:
        return PPOStats(
            loss_total=0.0, loss_ppo=0.0, loss_kl=0.0,
            approx_kl_new_old=0.0, approx_kl_new_ref=0.0, k2_ref_old=0.0,
            entropy=0.0, grad_norm=0.0,
            step_time_s=time.perf_counter()-t_start,
            train_policy_forwards=0, train_ref_forwards=0,
            train_policy_forward_time_s=0.0, train_ref_forward_time_s=0.0,
        )

    logp_old = batch.logp_old

    # Cached ref logp (optional)
    logp_ref_cached = None
    if variant == "A" and ref_compute == "cached":
        if batch.logp_ref is None:
            raise ValueError("ref_compute=cached but batch.logp_ref is None. Generate rollout with compute_ref_logp=True.")
        logp_ref_cached = batch.logp_ref

    fstats = ForwardStats()
    perm = torch.randperm(N, device=device)

    total_loss = total_ppo = total_kl = 0.0
    total_kl_new_old = total_kl_new_ref = total_k2 = 0.0
    steps = 0
    last_grad_norm = 0.0

    policy.train()

    for ep in range(ppo_epochs):
        # optional ref logp recompute per epoch (wasteful baseline)
        logp_ref_epoch = None
        if variant == "A" and ref_compute == "naive_epoch":
            if ref_policy is None:
                raise ValueError("variant A requires ref_policy")
            with torch.no_grad():
                logp_ref_epoch = _teacher_logp(ref_policy, batch.input_ids, fstats, "ref")

        for start in range(0, N, minibatch_size):
            mb = perm[start:start+minibatch_size]
            mb_idx = idx[mb]
            b_ix = mb_idx[:, 0]
            t_ix = mb_idx[:, 1]

            # Current logp under policy (teacher-forced).
            # Compute current logp under policy only for sequences touched by this minibatch.
            b_unique, inv = torch.unique(b_ix, return_inverse=True)
            input_mb = batch.input_ids[b_unique]
            logp_new_mb = _teacher_logp(policy, input_mb, fstats, "policy")
            lp_new = logp_new_mb[inv, t_ix]
            lp_old = logp_old[b_ix, t_ix]
            ratio = torch.exp(lp_new - lp_old)

            unclipped = ratio * adv_tok[b_ix, t_ix]
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high)
            clipped = clipped_ratio * adv_tok[b_ix, t_ix]
            loss_ppo = -torch.mean(torch.minimum(unclipped, clipped))

            if variant == "A":
                if ref_compute == "cached":
                    lp_ref = logp_ref_cached[b_ix, t_ix]
                else:
                    assert logp_ref_epoch is not None
                    lp_ref = logp_ref_epoch[b_ix, t_ix]

                kl_simple = torch.mean(lp_new - lp_ref)

                # k2 logging / optional penalty
                if k2_mode == "ref_over_old":
                    z = torch.clamp(lp_ref - lp_old, -10.0, 10.0)
                else:
                    z = torch.clamp(lp_new - lp_ref, -10.0, 10.0)
                k2 = 0.5 * (-z) ** 2
                k2_mean = torch.mean(k2)

                kl_pen = kl_simple
                if kl_estimator == "k2" and k2_mode == "theta_over_ref":
                    kl_pen = k2_mean

                loss_kl = beta_kl * kl_pen
                approx_kl_new_ref = float(kl_simple.detach().item())
                k2_val = float(k2_mean.detach().item())
            else:
                # Variant B: no reference; penalize divergence from old
                kl_old = torch.mean(lp_new - lp_old)
                loss_kl = beta_kl * kl_old
                approx_kl_new_ref = 0.0
                k2_val = 0.0

            approx_kl_new_old = float(torch.mean(lp_new - lp_old).detach().item())
            loss = loss_ppo + loss_kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            last_grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm).item())
            optimizer.step()

            total_loss += float(loss.detach().item())
            total_ppo += float(loss_ppo.detach().item())
            total_kl += float(loss_kl.detach().item())
            total_kl_new_old += approx_kl_new_old
            total_kl_new_ref += approx_kl_new_ref
            total_k2 += k2_val
            steps += 1

    step_time = time.perf_counter() - t_start
    denom = max(1, steps)
    return PPOStats(
        loss_total=total_loss/denom,
        loss_ppo=total_ppo/denom,
        loss_kl=total_kl/denom,
        approx_kl_new_old=total_kl_new_old/denom,
        approx_kl_new_ref=total_kl_new_ref/denom,
        k2_ref_old=total_k2/denom,
        entropy=0.0,
        grad_norm=last_grad_norm,
        step_time_s=step_time,
        train_policy_forwards=fstats.policy_forwards,
        train_ref_forwards=fstats.ref_forwards,
        train_policy_forward_time_s=fstats.policy_forward_time_s,
        train_ref_forward_time_s=fstats.ref_forward_time_s,
    )
