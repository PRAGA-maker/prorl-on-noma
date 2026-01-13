from __future__ import annotations

import argparse
import os
import time
import json
import random
from typing import Dict, Any

import torch

from .tokenizer import default_tokenizer
from .tasks import build_task
from .model import CausalGRULM
from .rollout import generate_rollout
from .ppo import ppo_update
from .logger import JSONLLogger
from .optim_utils import OptimizerSpec, create_adamw, hard_reset_optimizer, soft_washout_optimizer_state, name_to_param, migrate_optimizer_state_by_name

def _set_seed(seed: int) -> random.Random:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return rng

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scoped ProRL reset+KL benchmark")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)

    # core loop
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--task", type=str, default="reverse_digits")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=0, help="0 uses task default")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)

    # model
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--adapter_rank", type=int, default=0)
    p.add_argument("--adapter_scale", type=float, default=1.0)

    # PPO
    p.add_argument("--ppo_epochs", type=int, default=2)
    p.add_argument("--minibatch_size", type=int, default=2048)
    p.add_argument("--clip_eps_low", type=float, default=0.20)
    p.add_argument("--clip_eps_high", type=float, default=0.28)
    p.add_argument("--beta_kl", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Reference / KL
    p.add_argument("--variant", type=str, default="A", choices=["A", "B"])
    p.add_argument("--ref_compute", type=str, default="cached", choices=["cached", "naive_epoch"])
    p.add_argument("--kl_estimator", type=str, default="simple", choices=["simple", "k2"])
    p.add_argument("--k2_mode", type=str, default="ref_over_old", choices=["ref_over_old", "theta_over_ref"])
    p.add_argument("--reset_every", type=int, default=300)

    # Optimizer resets
    p.add_argument("--opt_reset", type=str, default="keep", choices=["keep", "hard", "soft", "asym"])
    p.add_argument("--soft_factor", type=float, default=0.2)
    p.add_argument("--asym_sq_factor", type=float, default=0.8)

    # Optional topology growth
    p.add_argument("--enable_growth", type=int, default=0)
    p.add_argument("--grow_every", type=int, default=500)
    p.add_argument("--rank_step", type=int, default=4)

    # Optimizer hyperparams
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_eps", type=float, default=1e-8)

    return p.parse_args()

def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    logger = JSONLLogger(args.outdir)
    rng = _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = default_tokenizer()
    task = build_task(args.task)
    max_new = task.answer_max_len if args.max_new_tokens == 0 else args.max_new_tokens

    # policy + reference
    policy = CausalGRULM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        adapter_rank=args.adapter_rank,
        adapter_scale=args.adapter_scale,
    ).to(device)

    ref = None
    if args.variant.upper() == "A":
        ref = CausalGRULM(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            adapter_rank=args.adapter_rank,
            adapter_scale=args.adapter_scale,
        ).to(device)
        ref.copy_from(policy)

    opt_spec = OptimizerSpec(
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )
    optimizer = create_adamw(policy, opt_spec)

    cfg: Dict[str, Any] = vars(args)
    cfg["device"] = str(device)
    cfg["tokenizer_vocab_size"] = tokenizer.vocab_size
    cfg["task_answer_max_len"] = task.answer_max_len
    logger.write_config(cfg)

    t_run0 = time.perf_counter()
    last_reset = 0
    last_grow = 0

    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()

        # reference resets (only variant A)
        if args.variant.upper() == "A" and args.reset_every > 0 and (step - last_reset) >= args.reset_every:
            assert ref is not None
            ref.copy_from(policy)
            logger.log_event({"step": step, "event": "ref_reset"})
            last_reset = step

            if args.opt_reset == "hard":
                optimizer = hard_reset_optimizer(policy, optimizer, opt_spec)
                logger.log_event({"step": step, "event": "opt_reset_hard"})
            elif args.opt_reset == "soft":
                soft_washout_optimizer_state(optimizer, factor=args.soft_factor, asym=False)
                logger.log_event({"step": step, "event": "opt_reset_soft", "factor": args.soft_factor})
            elif args.opt_reset == "asym":
                soft_washout_optimizer_state(optimizer, factor=args.soft_factor, asym=True, sq_factor=args.asym_sq_factor)
                logger.log_event({"step": step, "event": "opt_reset_asym", "factor": args.soft_factor, "sq_factor": args.asym_sq_factor})
            else:
                # keep
                pass

        # optional topology growth
        if args.enable_growth == 1 and args.grow_every > 0 and (step - last_grow) >= args.grow_every:
            old_named = name_to_param(policy)
            old_opt = optimizer
            growth_info = policy.maybe_grow_adapter(args.rank_step)
            if growth_info.get("changed", False):
                # recreate optimizer and migrate states by name (including overlapping slices for grown params)
                optimizer = create_adamw(policy, opt_spec)
                new_named = name_to_param(policy)
                migrate_optimizer_state_by_name(old_opt, optimizer, old_named, new_named)
                logger.log_event({
                    "step": step,
                    "event": "adapter_grow",
                    "growth": {k: v for k, v in growth_info.items() if k != "old_params" and k != "new_params"},
                })
            last_grow = step

        # rollout generation
        compute_ref_logp = (args.variant.upper() == "A" and args.ref_compute == "cached")
        rollout_batch, rollout_stats = generate_rollout(
            policy=policy,
            ref_policy=ref,
            tokenizer=tokenizer,
            task=task,
            batch_size=args.batch_size,
            rng=rng,
            device=device,
            max_new_tokens=max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            compute_ref_logp=compute_ref_logp,
        )

        # PPO update
        ppo_stats = ppo_update(
            policy=policy,
            ref_policy=ref,
            batch=rollout_batch,
            optimizer=optimizer,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            clip_eps_low=args.clip_eps_low,
            clip_eps_high=args.clip_eps_high,
            beta_kl=args.beta_kl,
            variant=args.variant,
            ref_compute=args.ref_compute,
            kl_estimator=args.kl_estimator,
            k2_mode=args.k2_mode,
            max_grad_norm=args.max_grad_norm,
        )

        # Metrics
        reward_mean = float(rollout_batch.rewards.mean().item())
        reward_rate = float((rollout_batch.rewards > 0.5).float().mean().item())
        gen_tokens = int(rollout_batch.action_mask.sum().item())
        total_tokens = int((rollout_batch.input_ids != tokenizer.pad_id).sum().item())

        wall = time.perf_counter() - t0
        throughput_tok_s = total_tokens / max(1e-8, wall)

        logger.log_metrics({
            "step": step,
            "reward_mean": reward_mean,
            "reward_rate": reward_rate,
            "gen_tokens": gen_tokens,
            "total_tokens": total_tokens,
            "throughput_tok_s": throughput_tok_s,

            "loss_total": ppo_stats.loss_total,
            "loss_ppo": ppo_stats.loss_ppo,
            "loss_kl": ppo_stats.loss_kl,
            "approx_kl_new_old": ppo_stats.approx_kl_new_old,
            "approx_kl_new_ref": ppo_stats.approx_kl_new_ref,
            "k2_ref_old": ppo_stats.k2_ref_old,
            "grad_norm": ppo_stats.grad_norm,

            "rollout_policy_forwards": rollout_stats.policy_forwards,
            "rollout_ref_forwards": rollout_stats.ref_forwards,
            "rollout_policy_forward_time_s": rollout_stats.policy_forward_time_s,
            "rollout_ref_forward_time_s": rollout_stats.ref_forward_time_s,

            "train_policy_forwards": ppo_stats.train_policy_forwards,
            "train_ref_forwards": ppo_stats.train_ref_forwards,
            "train_policy_forward_time_s": ppo_stats.train_policy_forward_time_s,
            "train_ref_forward_time_s": ppo_stats.train_ref_forward_time_s,

            "step_wall_time_s": wall,
            "run_wall_time_s": time.perf_counter() - t_run0,
        })

        # lightweight stdout every 50
        if step % 50 == 0 or step == 1:
            print(f"[{step:05d}] reward_mean={reward_mean:.3f} rate={reward_rate:.3f} "
                  f"KL_ref={ppo_stats.approx_kl_new_ref:.4f} KL_old={ppo_stats.approx_kl_new_old:.4f} "
                  f"roll_ref_fw={rollout_stats.ref_forwards} train_ref_fw={ppo_stats.train_ref_forwards} "
                  f"tok/s={throughput_tok_s:.1f}")

    print("Done. Wrote:", args.outdir)

if __name__ == "__main__":
    main()
