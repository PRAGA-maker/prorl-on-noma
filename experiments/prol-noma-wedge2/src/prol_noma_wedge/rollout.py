from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time
import torch
import torch.nn.functional as F

from .task import make_example, reward_exact
from .tokenizer import CharTokenizer
from .model import GRULM, logprobs_from_logits

@dataclass
class RolloutBatch:
    input_ids: torch.Tensor      # (B, T_full)
    prompt_T: int
    answer_T: int
    logp_old: torch.Tensor       # (B, answer_T)
    ref_logp: Optional[torch.Tensor]  # (B, answer_T) or None
    rewards: torch.Tensor        # (B,)
    truth_answers: list          # list[str]
    pred_answers: list           # list[str]

@torch.no_grad()
def rollout_batch(
    policy: GRULM,
    ref: Optional[GRULM],
    tok: CharTokenizer,
    task: str,
    batch_size: int,
    prompt_len: int,
    answer_len: int,
    device: torch.device,
    temperature: float = 1.0,
    compute_ref_logp: bool = True,
) -> RolloutBatch:
    policy.eval()
    if ref is not None:
        ref.eval()

    prompts, truths = [], []
    for _ in range(batch_size):
        p, a = make_example(task, prompt_len=prompt_len, answer_len=answer_len)
        prompts.append(p)
        truths.append(a)

    # Tokenize prompt with BOS
    prompt_ids = [tok.encode(p, add_bos=True, add_eos=False) for p in prompts]
    prompt_T = len(prompt_ids[0])
    assert all(len(x) == prompt_T for x in prompt_ids), "Prompt length must be constant in this toy setup."

    # Initialize sequences
    B = batch_size
    answer_T = answer_len
    T_full = prompt_T + answer_T

    input_ids = torch.full((B, T_full), fill_value=tok.pad_id, dtype=torch.long, device=device)
    for i in range(B):
        input_ids[i, :prompt_T] = torch.tensor(prompt_ids[i], device=device, dtype=torch.long)

    # Autoregressive sample answer tokens
    logp_old = torch.zeros((B, answer_T), device=device)
    for t in range(answer_T):
        cur = input_ids[:, :prompt_T + t]  # (B, prompt_T+t)
        logits = policy(cur)               # (B, curT, V)
        next_logits = logits[:, -1, :] / max(1e-6, temperature)
        probs = F.softmax(next_logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
        # logp of sampled token
        logp = torch.log(torch.gather(probs, 1, next_tok.unsqueeze(1)).squeeze(1).clamp_min(1e-12))
        logp_old[:, t] = logp
        input_ids[:, prompt_T + t] = next_tok

    # Decode answers + rewards
    pred_answers = []
    rewards = torch.zeros((B,), device=device)
    for i in range(B):
        ans_ids = input_ids[i, prompt_T:prompt_T+answer_T].tolist()
        pred = tok.decode(ans_ids)
        pred_answers.append(pred)
        rewards[i] = reward_exact(pred, truths[i])

    # Compute ref_logp once for the batch (teacher-forcing) if requested
    ref_logp = None
    if ref is not None and compute_ref_logp:
        ref_logp = compute_answer_logprobs(ref, input_ids, prompt_T=prompt_T, answer_T=answer_T)

    return RolloutBatch(
        input_ids=input_ids,
        prompt_T=prompt_T,
        answer_T=answer_T,
        logp_old=logp_old,
        ref_logp=ref_logp,
        rewards=rewards,
        truth_answers=truths,
        pred_answers=pred_answers,
    )

def compute_answer_logprobs(model: GRULM, input_ids_full: torch.Tensor, prompt_T: int, answer_T: int) -> torch.Tensor:
    # Teacher-forcing logprobs for the answer tokens.
    # logits at position i predict token at position i+1.
    logits = model(input_ids_full)  # (B, T, V)
    logp = F.log_softmax(logits, dim=-1)
    targets = input_ids_full[:, 1:]  # (B, T-1) tokens to predict
    pred_logp = torch.gather(logp[:, :-1, :], dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    start = prompt_T - 1
    return pred_logp[:, start:start + answer_T]
