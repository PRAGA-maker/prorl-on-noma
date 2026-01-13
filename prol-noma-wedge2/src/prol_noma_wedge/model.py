from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 2
    dropout: float = 0.1

class GRULM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.gru = nn.GRU(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            num_layers=cfg.n_layers,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T)
        x = self.emb(input_ids)  # (B, T, D)
        h, _ = self.gru(x)       # (B, T, D)
        h = self.ln(h)
        logits = self.head(h)    # (B, T, V)
        return logits

    @torch.no_grad()
    def clone_frozen(self) -> "GRULM":
        m = copy.deepcopy(self)
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
        return m

def logprobs_from_logits(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    # logits: (B, T, V), actions: (B, T) token ids
    logp = F.log_softmax(logits, dim=-1)
    return torch.gather(logp, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
