from __future__ import annotations
import argparse
from pathlib import Path

from .config import load_yaml, deep_update
from .trainer import TrainConfig, train

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[2] / "configs" / "default.yaml"))
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--variant", type=str, choices=["A","B","C"], default=None)
    ap.add_argument("--task", type=str, choices=["reverse_digits","parity"], default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--update_epochs", type=int, default=None)
    ap.add_argument("--minibatch_size", type=int, default=None)
    ap.add_argument("--beta_kl", type=float, default=None)
    ap.add_argument("--reset_every", type=int, default=None)
    ap.add_argument("--optimizer_reset", type=str, choices=["keep","hard","soft"], default=None)
    ap.add_argument("--soft_reset_factor", type=float, default=None)
    ap.add_argument("--naive_ref_each_epoch", action="store_true")
    ap.add_argument("--device", type=str, default=None)

    return ap.parse_args()

def main() -> None:
    args = parse_args()
    base = load_yaml(args.config)

    # Apply CLI overrides
    overrides = {"algo": {}, "model": {}, "optim": {}}
    if args.variant: overrides["algo"]["variant"] = args.variant
    if args.task: base["task"] = args.task
    if args.steps is not None: base["steps"] = args.steps
    if args.batch_size is not None: base["batch_size"] = args.batch_size
    if args.update_epochs is not None: overrides["algo"]["update_epochs"] = args.update_epochs
    if args.minibatch_size is not None: overrides["algo"]["minibatch_size"] = args.minibatch_size
    if args.beta_kl is not None: overrides["algo"]["beta_kl"] = args.beta_kl
    if args.reset_every is not None: overrides["algo"]["reset_every"] = args.reset_every
    if args.optimizer_reset is not None: overrides["algo"]["optimizer_reset"] = args.optimizer_reset
    if args.soft_reset_factor is not None: overrides["algo"]["soft_reset_factor"] = args.soft_reset_factor
    if args.naive_ref_each_epoch: overrides["algo"]["naive_ref_each_epoch"] = True
    if args.device is not None: base["device"] = args.device

    cfg_dict = deep_update(base, overrides)

    cfg = TrainConfig(
        seed=int(cfg_dict.get("seed", 0)),
        device=str(cfg_dict.get("device", "cpu")),
        task=str(cfg_dict.get("task", "reverse_digits")),
        steps=int(cfg_dict.get("steps", 1000)),
        batch_size=int(cfg_dict.get("batch_size", 64)),
        prompt_len=int(cfg_dict.get("prompt_len", 12)),
        answer_len=int(cfg_dict.get("answer_len", 12)),

        d_model=int(cfg_dict.get("model", {}).get("d_model", 128)),
        n_layers=int(cfg_dict.get("model", {}).get("n_layers", 2)),
        dropout=float(cfg_dict.get("model", {}).get("dropout", 0.1)),

        lr=float(cfg_dict.get("optim", {}).get("lr", 3e-4)),
        betas=tuple(cfg_dict.get("optim", {}).get("betas", [0.9, 0.999])),
        weight_decay=float(cfg_dict.get("optim", {}).get("weight_decay", 0.0)),

        variant=str(cfg_dict.get("algo", {}).get("variant", "A")),
        beta_kl=float(cfg_dict.get("algo", {}).get("beta_kl", 0.02)),
        update_epochs=int(cfg_dict.get("algo", {}).get("update_epochs", 4)),
        minibatch_size=int(cfg_dict.get("algo", {}).get("minibatch_size", 32)),
        clip_ratio=float(cfg_dict.get("algo", {}).get("clip_ratio", 0.2)),
        baseline_ema=float(cfg_dict.get("algo", {}).get("baseline_ema", 0.95)),

        reset_every=int(cfg_dict.get("algo", {}).get("reset_every", 400)),
        optimizer_reset=str(cfg_dict.get("algo", {}).get("optimizer_reset", "keep")),
        soft_reset_factor=float(cfg_dict.get("algo", {}).get("soft_reset_factor", 0.2)),

        naive_ref_each_epoch=bool(cfg_dict.get("algo", {}).get("naive_ref_each_epoch", False)),
    )

    train(outdir=args.outdir, cfg=cfg)

if __name__ == "__main__":
    main()
