from __future__ import annotations
import json, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@dataclass
class Counters:
    policy_forwards: int = 0
    ref_forwards: int = 0

class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", encoding="utf-8")

    def log(self, row: Dict[str, Any]) -> None:
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass

def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}

def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-8) -> torch.Tensor:
    # mask is 0/1
    x = x * mask
    denom = mask.sum(dim=dim).clamp_min(eps)
    return x.sum(dim=dim) / denom
