from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random

@dataclass
class TaskSpec:
    name: str
    prompt_len: int
    answer_len: int

def _rand_digits(n: int) -> str:
    return "".join(random.choice("0123456789") for _ in range(n))

def make_example(task: str, prompt_len: int, answer_len: int) -> Tuple[str, str]:
    # prompt_len/answer_len here refer to the digit payload length, not the full prompt string.
    digits = _rand_digits(prompt_len)
    if task == "reverse_digits":
        prompt = f"rev:{digits}->"
        answer = digits[::-1][:answer_len]
    elif task == "parity":
        prompt = f"par:{digits}->"
        s = sum(int(c) for c in digits)
        answer = "even" if (s % 2 == 0) else "odd"
        # pad/crop answer to answer_len (this is a toy)
        if len(answer) < answer_len:
            answer = answer + (" " * (answer_len - len(answer)))
        else:
            answer = answer[:answer_len]
    else:
        raise ValueError(f"Unknown task: {task}")
    return prompt, answer

def reward_exact(pred: str, truth: str) -> float:
    return 1.0 if pred == truth else 0.0
