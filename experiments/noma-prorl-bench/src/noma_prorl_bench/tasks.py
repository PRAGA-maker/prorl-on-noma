from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable
import random

@dataclass
class Batch:
    prompts: List[str]
    targets: List[str]

class Task:
    name: str
    prompt_prefix: str
    answer_max_len: int

    def sample_batch(self, batch_size: int, rng: random.Random) -> Batch:
        raise NotImplementedError

    def score(self, prompt: str, target: str, completion: str) -> float:
        """Return reward in [0,1]. Default: exact match on target (trimmed)."""
        return 1.0 if completion.strip() == target.strip() else 0.0

class ReverseDigits(Task):
    name = "reverse_digits"
    prompt_prefix = "REV:"
    answer_max_len = 12

    def __init__(self, min_len: int = 3, max_len: int = 8):
        self.min_len = min_len
        self.max_len = max_len

    def sample_batch(self, batch_size: int, rng: random.Random) -> Batch:
        prompts, targets = [], []
        for _ in range(batch_size):
            n = rng.randint(self.min_len, self.max_len)
            s = "".join(str(rng.randint(0, 9)) for _ in range(n))
            prompts.append(f"{self.prompt_prefix} {s} = ")
            targets.append(s[::-1])
        return Batch(prompts, targets)

class Parity(Task):
    name = "parity"
    prompt_prefix = "PAR:"
    answer_max_len = 1

    def __init__(self, min_len: int = 3, max_len: int = 10):
        self.min_len = min_len
        self.max_len = max_len

    def sample_batch(self, batch_size: int, rng: random.Random) -> Batch:
        prompts, targets = [], []
        for _ in range(batch_size):
            n = rng.randint(self.min_len, self.max_len)
            digits = [rng.randint(0, 9) for _ in range(n)]
            s = "".join(str(d) for d in digits)
            parity = sum(digits) % 2
            prompts.append(f"{self.prompt_prefix} {s} = ")
            targets.append("1" if parity == 1 else "0")
        return Batch(prompts, targets)

class AddMod10(Task):
    name = "add_mod10"
    prompt_prefix = "ADD:"
    answer_max_len = 1

    def sample_batch(self, batch_size: int, rng: random.Random) -> Batch:
        prompts, targets = [], []
        for _ in range(batch_size):
            a = rng.randint(0, 99)
            b = rng.randint(0, 99)
            prompts.append(f"{self.prompt_prefix} {a}+{b} mod10 = ")
            targets.append(str((a + b) % 10))
        return Batch(prompts, targets)

def build_task(name: str) -> Task:
    name = name.strip().lower()
    if name == ReverseDigits.name:
        return ReverseDigits()
    if name == Parity.name:
        return Parity()
    if name == AddMod10.name:
        return AddMod10()
    raise ValueError(f"Unknown task {name}. Options: reverse_digits, parity, add_mod10")
