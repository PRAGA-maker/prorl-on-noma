from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch
import torch.nn.functional as F

from .model import GRULM
from .rollout import compute_answer_logprobs

@dataclass
class LossOut:
    loss: torch.Tensor
    pg_loss: torch.Tensor
    kl_mean: torch.Tensor
    entropy_mean: torch.Tensor

def ppo_token_loss(
    logp_new: torch.Tensor,          # (B, T)
    logp_old: torch.Tensor,          # (B, T)
    advantages: torch.Tensor,        # (B,)
    clip_ratio: float,
) -> torch.Tensor:
    # Broadcast advantages to token positions
    adv = advantages.unsqueeze(1)  # (B,1)
    ratio = torch.exp(logp_new - logp_old)
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    obj = torch.minimum(ratio * adv, clipped * adv)
    return -obj.mean()

def entropy_from_logp_sampled(logp_new: torch.Tensor) -> torch.Tensor:
    # crude proxy: entropy lower-bounded by negative mean logp of sampled tokens
    return (-logp_new).mean()

def compute_losses(
    policy: GRULM,
    ref: Optional[GRULM],
    input_ids: torch.Tensor,
    prompt_T: int,
    answer_T: int,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    beta_kl: float,
    clip_ratio: float,
    variant: str,
    cached_ref_logp: Optional[torch.Tensor] = None,
    naive_ref_each_epoch: bool = False,
) -> LossOut:
    # Compute new logprobs under current policy
    logp_new = compute_answer_logprobs(policy, input_ids, prompt_T=prompt_T, answer_T=answer_T)

    pg_loss = ppo_token_loss(logp_new=logp_new, logp_old=logp_old, advantages=advantages, clip_ratio=clip_ratio)

    if variant in ("A", "C"):
        if ref is None:
            raise ValueError("Variant A/C requires a reference model.")
        if naive_ref_each_epoch:
            # compute ref logp now
            with torch.no_grad():
                logp_ref = compute_answer_logprobs(ref, input_ids, prompt_T=prompt_T, answer_T=answer_T)
        else:
            if cached_ref_logp is None:
                raise ValueError("cached_ref_logp required unless naive_ref_each_epoch=True")
            logp_ref = cached_ref_logp
        kl_est = (logp_new - logp_ref)
        kl_mean = kl_est.mean()
        loss = pg_loss + beta_kl * kl_mean
    elif variant == "B":
        # Behavior-policy trust region (no reference model)
        kl_est = (logp_new - logp_old)
        kl_mean = kl_est.mean()
        loss = pg_loss + beta_kl * kl_mean
    else:
        raise ValueError(f"Unknown variant: {variant}")

    ent = entropy_from_logp_sampled(logp_new)
    return LossOut(loss=loss, pg_loss=pg_loss.detach(), kl_mean=kl_mean.detach(), entropy_mean=ent.detach())
