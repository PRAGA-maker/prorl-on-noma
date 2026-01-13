from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import torch

@dataclass
class OptimizerSpec:
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    eps: float = 1e-8

def create_adamw(model: torch.nn.Module, spec: OptimizerSpec) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=spec.lr, betas=spec.betas, weight_decay=spec.weight_decay, eps=spec.eps)

def hard_reset_optimizer(model: torch.nn.Module, old_opt: torch.optim.Optimizer, spec: OptimizerSpec) -> torch.optim.Optimizer:
    """Drop optimizer state completely (paper-like hard reset)."""
    return create_adamw(model, spec)

def soft_washout_optimizer_state(opt: torch.optim.Optimizer, factor: float, *, asym: bool = False, sq_factor: Optional[float] = None) -> None:
    """Decay Adam moments in-place without recreating optimizer.

    factor: multiplier for exp_avg (and exp_avg_sq unless asym/sq_factor specified)
    """
    if sq_factor is None:
        sq_factor = factor
    for param, st in opt.state.items():
        if not isinstance(st, dict):
            continue
        if "exp_avg" in st and torch.is_tensor(st["exp_avg"]):
            st["exp_avg"].mul_(factor)
        if "exp_avg_sq" in st and torch.is_tensor(st["exp_avg_sq"]):
            st["exp_avg_sq"].mul_(sq_factor)

def name_to_param(model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    return {n: p for n, p in model.named_parameters()}

def _copy_overlap(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy overlapping leading slices from src into dst."""
    if dst.shape == src.shape:
        dst.copy_(src)
        return
    # Support 1D/2D/3D tensors for this toy repo.
    slices = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape))
    dst[slices].copy_(src[slices])

def migrate_optimizer_state_by_name(
    old_opt: torch.optim.Optimizer,
    new_opt: torch.optim.Optimizer,
    old_named: Dict[str, torch.nn.Parameter],
    new_named: Dict[str, torch.nn.Parameter],
) -> None:
    """Migrate Adam-like optimizer state across a model mutation (e.g. adapter growth).

    Strategy:
      - match params by name
      - copy scalar state fields (e.g. step)
      - for tensor fields, copy overlapping slices (top-left)
    """
    for name, new_p in new_named.items():
        old_p = old_named.get(name)
        if old_p is None:
            continue
        old_state = old_opt.state.get(old_p)
        if not isinstance(old_state, dict) or len(old_state) == 0:
            continue
        st = new_opt.state.setdefault(new_p, {})
        for k, v in old_state.items():
            if torch.is_tensor(v):
                st[k] = torch.zeros_like(new_p.data)
                _copy_overlap(st[k], v)
            else:
                # step is int; copy directly
                st[k] = v
