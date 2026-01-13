from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import time
import torch
import torch.nn.functional as F

from .tokenizer import CharTokenizer, BOS, EOS
from .tasks import Task

@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [B, T]
    action_mask: torch.Tensor        # [B, T] bool (1 where token was generated)
    logp_old: torch.Tensor           # [B, T]
    logp_ref: Optional[torch.Tensor] # [B, T] or None
    rewards: torch.Tensor            # [B]
    prompts: List[str]
    targets: List[str]
    completions: List[str]

@dataclass
class ForwardStats:
    policy_forwards: int = 0
    ref_forwards: int = 0
    policy_forward_time_s: float = 0.0
    ref_forward_time_s: float = 0.0

def _forward(model, input_ids, hidden, stats: ForwardStats, which: str):
    t0 = time.perf_counter()
    logits, hidden2 = model(input_ids, hidden)
    dt = time.perf_counter() - t0
    if which == "policy":
        stats.policy_forwards += 1
        stats.policy_forward_time_s += dt
    else:
        stats.ref_forwards += 1
        stats.ref_forward_time_s += dt
    return logits, hidden2

@torch.no_grad()
def generate_rollout(
    policy,
    ref_policy,
    tokenizer: CharTokenizer,
    task: Task,
    batch_size: int,
    rng,
    device: torch.device,
    max_new_tokens: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    compute_ref_logp: bool = True,
) -> Tuple[RolloutBatch, ForwardStats]:
    """Generate rollouts using the policy (as behavior policy).

    Returns (RolloutBatch, ForwardStats). The returned logp_old is under behavior policy.
    If compute_ref_logp is True and ref_policy is not None, logp_ref is computed on the fly
    for generated tokens (teacher forcing on generated sequence).
    """
    policy.eval()
    if ref_policy is not None:
        ref_policy.eval()

    stats = ForwardStats()
    batch = task.sample_batch(batch_size, rng)
    prompts, targets = batch.prompts, batch.targets

    if max_new_tokens is None:
        max_new_tokens = task.answer_max_len

    # Encode prompts
    prompt_ids_list: List[List[int]] = []
    for p in prompts:
        ids = tokenizer.encode(p, add_bos=True, add_eos=False)
        prompt_ids_list.append(ids)

    # Pad prompts to same length
    max_prompt = max(len(x) for x in prompt_ids_list)
    B = batch_size
    prompt_ids = torch.full((B, max_prompt), tokenizer.pad_id, dtype=torch.long, device=device)
    prompt_lens = torch.tensor([len(x) for x in prompt_ids_list], device=device, dtype=torch.long)
    for i, ids in enumerate(prompt_ids_list):
        prompt_ids[i, :len(ids)] = torch.tensor(ids, device=device)

    # Run prompt through policy to get hidden state at end of prompt.
    # We process prompt in one forward; then generate token-by-token incrementally.
    logits, hidden = _forward(policy, prompt_ids, None, stats, "policy")
    # Get last non-pad position logits for each sample:
    last_idx = (prompt_lens - 1).clamp(min=0)
    last_logits = logits[torch.arange(B, device=device), last_idx]  # [B, V]

    # We'll build sequences as python lists then pad.
    seq_ids: List[List[int]] = [prompt_ids_list[i].copy() for i in range(B)]
    logp_old_list: List[List[float]] = [[0.0]*len(seq_ids[i]) for i in range(B)]  # prompt positions set to 0
    action_mask_list: List[List[int]] = [[0]*len(seq_ids[i]) for i in range(B)]  # prompt positions 0
    # Keep per-sample hidden state for incremental generation:
    # hidden: [n_layers, B, d]
    done = torch.zeros(B, device=device, dtype=torch.bool)

    for _t in range(max_new_tokens):
        # sample next token from last_logits
        logits_t = last_logits / max(1e-8, temperature)
        if top_k and top_k > 0:
            topv, topi = torch.topk(logits_t, k=min(top_k, logits_t.size(-1)), dim=-1)
            mask = torch.full_like(logits_t, float("-inf"))
            mask.scatter_(1, topi, topv)
            logits_t = mask

        probs = F.softmax(logits_t, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
        # compute logp under behavior policy
        logp = F.log_softmax(last_logits, dim=-1).gather(1, next_token[:, None]).squeeze(1)

        # append tokens
        for i in range(B):
            if done[i]:
                continue
            tok = int(next_token[i].item())
            seq_ids[i].append(tok)
            logp_old_list[i].append(float(logp[i].item()))
            action_mask_list[i].append(1)
            if tok == tokenizer.eos_id:
                done[i] = True

        if bool(done.all()):
            break

        # incremental forward: feed next_token as input with hidden
        # shape [B,1]
        inp = next_token[:, None]
        logits2, hidden = _forward(policy, inp, hidden, stats, "policy")
        last_logits = logits2[:, -1, :]

    # Pad sequences to same length
    max_T = max(len(s) for s in seq_ids)
    input_ids = torch.full((B, max_T), tokenizer.pad_id, dtype=torch.long, device=device)
    action_mask = torch.zeros((B, max_T), dtype=torch.bool, device=device)
    logp_old = torch.zeros((B, max_T), dtype=torch.float32, device=device)
    for i in range(B):
        L = len(seq_ids[i])
        input_ids[i, :L] = torch.tensor(seq_ids[i], dtype=torch.long, device=device)
        action_mask[i, :L] = torch.tensor(action_mask_list[i], dtype=torch.bool, device=device)
        logp_old[i, :L] = torch.tensor(logp_old_list[i], dtype=torch.float32, device=device)

    # Decode completions: tokens after prompt length, stopping at eos
    completions: List[str] = []
    for i in range(B):
        prompt_len = len(prompt_ids_list[i])
        comp_ids = seq_ids[i][prompt_len:]
        completions.append(tokenizer.decode(comp_ids, stop_at_eos=True))

    # Rewards
    rewards = torch.tensor([task.score(prompts[i], targets[i], completions[i]) for i in range(B)],
                           dtype=torch.float32, device=device)

    logp_ref = None
    if compute_ref_logp and ref_policy is not None:
        # Teacher forcing through ref model on the full sequence to get per-token logprobs.
        # We only need logprobs for generated tokens.
        # Compute logits for all positions, then logp for token at each position.
        # For token y_t, the model predicts it at position t-1 (next-token).
        # We'll align by shifting.
        # input_ids: [B,T]
        logits_ref, _ = _forward(ref_policy, input_ids, None, stats, "ref")  # [B,T,V]
        logprobs_ref = F.log_softmax(logits_ref, dim=-1)
        # gather logp of the actual next token for each position
        # shift: prediction at t predicts token at t
        # so we gather token at t from logprobs at t-1; for t=0 invalid -> leave 0
        logp_ref = torch.zeros((B, max_T), dtype=torch.float32, device=device)
        # positions 1..T-1
        next_tokens = input_ids[:, 1:]  # [B,T-1]
        lp = logprobs_ref[:, :-1, :].gather(2, next_tokens[:, :, None]).squeeze(2)  # [B,T-1]
        logp_ref[:, 1:] = lp

        # For prompt tokens, action_mask is 0; downstream will ignore.
    return RolloutBatch(
        input_ids=input_ids,
        action_mask=action_mask,
        logp_old=logp_old,
        logp_ref=logp_ref,
        rewards=rewards,
        prompts=prompts,
        targets=targets,
        completions=completions,
    ), stats
