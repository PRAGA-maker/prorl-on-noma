from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankAdapter(nn.Module):
    """Residual low-rank adapter: h + scale * (A @ (B @ h)).

    Shapes:
      B: [rank, d_model]
      A: [d_model, rank]
    """
    def __init__(self, d_model: int, rank: int, scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.scale = scale
        # small init so adapter starts near zero
        self.B = nn.Parameter(torch.empty(rank, d_model))
        self.A = nn.Parameter(torch.empty(d_model, rank))
        nn.init.normal_(self.B, mean=0.0, std=0.02)
        nn.init.normal_(self.A, mean=0.0, std=0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, D]
        if self.rank == 0:
            return h
        # (B@h): [B, T, R]
        x = torch.einsum("rd,btd->btr", self.B, h)
        # (A@x): [B, T, D]
        y = torch.einsum("dr,btr->btd", self.A, x)
        return h + self.scale * y

    @torch.no_grad()
    def grow(self, rank_step: int) -> Dict[str, Any]:
        """Increase rank by rank_step, preserving existing weights.

        Returns a dict describing how parameters changed, useful for optimizer-state migration.
        """
        if rank_step <= 0:
            return {"changed": False}
        old_rank = self.rank
        new_rank = old_rank + rank_step
        old_B = self.B
        old_A = self.A

        # Create new params
        new_B = nn.Parameter(torch.empty(new_rank, self.d_model, device=old_B.device, dtype=old_B.dtype))
        new_A = nn.Parameter(torch.empty(self.d_model, new_rank, device=old_A.device, dtype=old_A.dtype))
        nn.init.normal_(new_B, mean=0.0, std=0.02)
        nn.init.normal_(new_A, mean=0.0, std=0.02)

        # Copy old slices
        new_B[:old_rank].copy_(old_B)
        new_A[:, :old_rank].copy_(old_A)

        # Swap in
        self.rank = new_rank
        self.B = new_B
        self.A = new_A

        return {
            "changed": True,
            "old_rank": old_rank,
            "new_rank": new_rank,
            "old_params": {"B": old_B, "A": old_A},
            "new_params": {"B": self.B, "A": self.A},
        }

class CausalGRULM(nn.Module):
    """Tiny causal language model for toy tasks."""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 1,
        dropout: float = 0.0,
        adapter_rank: int = 0,
        adapter_scale: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.adapter = LowRankAdapter(d_model, adapter_rank, scale=adapter_scale) if adapter_rank > 0 else None
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # init
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_ids: [B, T]
        x = self.embed(input_ids)
        h, hn = self.gru(x, hidden)
        if self.adapter is not None:
            h = self.adapter(h)
        logits = self.lm_head(h)
        return logits, hn

    @torch.no_grad()
    def copy_from(self, other: "CausalGRULM") -> None:
        """Copy weights from another model with identical architecture."""
        self.load_state_dict(other.state_dict(), strict=True)

    @torch.no_grad()
    def maybe_grow_adapter(self, rank_step: int) -> Dict[str, Any]:
        """Grow adapter rank if present; if absent and rank_step>0, create adapter."""
        if rank_step <= 0:
            return {"changed": False}
        if self.adapter is None:
            self.adapter = LowRankAdapter(self.d_model, rank_step, scale=1.0).to(next(self.parameters()).device)
            return {
                "changed": True,
                "created": True,
                "old_params": {},
                "new_params": {"B": self.adapter.B, "A": self.adapter.A},
                "old_rank": 0,
                "new_rank": rank_step,
            }
        else:
            return self.adapter.grow(rank_step)
