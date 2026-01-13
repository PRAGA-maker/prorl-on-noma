from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import time
import torch

from .model import GRULM, ModelConfig
from .tokenizer import CharTokenizer
from .rollout import rollout_batch, compute_answer_logprobs
from .algos import compute_losses
from .utils import set_seed, JsonlLogger, save_json, Counters

@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cpu"

    task: str = "reverse_digits"
    steps: int = 1000
    batch_size: int = 32
    prompt_len: int = 12
    answer_len: int = 12

    d_model: int = 64
    n_layers: int = 1
    dropout: float = 0.1

    lr: float = 3e-4
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.0

    variant: str = "A"              # A|B|C
    beta_kl: float = 0.02
    update_epochs: int = 2
    minibatch_size: int = 16
    clip_ratio: float = 0.2
    baseline_ema: float = 0.95

    reset_every: int = 400
    optimizer_reset: str = "keep"   # keep|hard|soft
    soft_reset_factor: float = 0.2

    naive_ref_each_epoch: bool = False
    temperature: float = 1.0

def _build_optimizer(policy: GRULM, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(policy.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)

def _soft_reset_adam(optim: torch.optim.Optimizer, factor: float) -> None:
    # Scale Adam moments in-place
    for group in optim.param_groups:
        for p in group["params"]:
            st = optim.state.get(p, None)
            if not st:
                continue
            for k in ("exp_avg", "exp_avg_sq"):
                if k in st and isinstance(st[k], torch.Tensor):
                    st[k].mul_(factor)

def train(outdir: str | Path, cfg: TrainConfig) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    save_json(outdir / "config.json", cfg.__dict__)

    set_seed(cfg.seed)
    torch.set_num_threads(1)
    device = torch.device(cfg.device)

    tok = CharTokenizer.build()
    policy = GRULM(ModelConfig(vocab_size=len(tok.stoi), d_model=cfg.d_model, n_layers=cfg.n_layers, dropout=cfg.dropout)).to(device)
    policy.train()

    ref = None
    if cfg.variant in ("A", "C"):
        ref = policy.clone_frozen().to(device)

    optim = _build_optimizer(policy, cfg)

    logger = JsonlLogger(outdir / "train.jsonl")
    counters = Counters()

    baseline = 0.0  # EMA baseline

    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        step_start = time.time()

        # Periodic reference reset
        if ref is not None and cfg.reset_every > 0 and (step % cfg.reset_every == 0):
            ref = policy.clone_frozen().to(device)
            # optimizer reset semantics
            if cfg.optimizer_reset == "hard":
                optim = _build_optimizer(policy, cfg)
            elif cfg.optimizer_reset == "soft":
                _soft_reset_adam(optim, cfg.soft_reset_factor)
            elif cfg.optimizer_reset == "keep":
                pass
            else:
                raise ValueError(f"Unknown optimizer_reset: {cfg.optimizer_reset}")

        # Rollout
        # If naive_ref_each_epoch, skip cached ref_logp computation to mimic costly baseline.
        compute_ref = (ref is not None) and (not cfg.naive_ref_each_epoch)
        batch = rollout_batch(
            policy=policy,
            ref=ref,
            tok=tok,
            task=cfg.task,
            batch_size=cfg.batch_size,
            prompt_len=cfg.prompt_len,
            answer_len=cfg.answer_len,
            device=device,
            temperature=cfg.temperature,
            compute_ref_logp=compute_ref,
        )
        counters.policy_forwards += cfg.answer_len  # autoregressive rollout does answer_len forwards
        if ref is not None and compute_ref:
            counters.ref_forwards += 1  # teacher-forcing pass over full seq

        rewards = batch.rewards.detach()
        r_mean = float(rewards.mean().item())

        # Update baseline
        baseline = cfg.baseline_ema * baseline + (1.0 - cfg.baseline_ema) * r_mean
        adv = (rewards - baseline).detach()

        # PPO-like multiple update epochs with minibatches
        B = cfg.batch_size
        idx = torch.arange(B, device=device)

        pg_loss_val, kl_val, ent_val, total_loss_val = 0.0, 0.0, 0.0, 0.0
        for ep in range(cfg.update_epochs):
            perm = idx[torch.randperm(B)]
            for start in range(0, B, cfg.minibatch_size):
                mb = perm[start:start+cfg.minibatch_size]
                mb_input = batch.input_ids[mb]
                mb_logp_old = batch.logp_old[mb]
                mb_adv = adv[mb]

                mb_ref_logp = None
                if cfg.variant in ("A", "C") and (not cfg.naive_ref_each_epoch):
                    mb_ref_logp = batch.ref_logp[mb] if batch.ref_logp is not None else None

                losses = compute_losses(
                    policy=policy,
                    ref=ref,
                    input_ids=mb_input,
                    prompt_T=batch.prompt_T,
                    answer_T=batch.answer_T,
                    logp_old=mb_logp_old,
                    advantages=mb_adv,
                    beta_kl=cfg.beta_kl,
                    clip_ratio=cfg.clip_ratio,
                    variant=cfg.variant,
                    cached_ref_logp=mb_ref_logp,
                    naive_ref_each_epoch=cfg.naive_ref_each_epoch,
                )
                counters.policy_forwards += 1  # teacher-forcing forward on policy
                if cfg.variant in ("A", "C") and cfg.naive_ref_each_epoch:
                    counters.ref_forwards += 1  # teacher-forcing forward on ref

                optim.zero_grad(set_to_none=True)
                losses.loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optim.step()

                pg_loss_val = float(losses.pg_loss.item())
                kl_val = float(losses.kl_mean.item())
                ent_val = float(losses.entropy_mean.item())
                total_loss_val = float(losses.loss.item())

        logger.log({
            "step": step,
            "reward_mean": r_mean,
            "baseline": float(baseline),
            "pg_loss": pg_loss_val,
            "kl_mean": kl_val,
            "entropy_proxy": ent_val,
            "loss": total_loss_val,
            "policy_forwards": counters.policy_forwards,
            "ref_forwards": counters.ref_forwards,
            "sec_step": time.time() - step_start,
            "sec_total": time.time() - t0,
        })

    logger.close()
