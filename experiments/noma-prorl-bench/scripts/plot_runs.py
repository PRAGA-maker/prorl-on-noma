#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict

import matplotlib.pyplot as plt

def read_metrics(path: str) -> List[Dict]:
    metrics_path = os.path.join(path, "metrics.jsonl")
    out = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="List of run directories")
    ap.add_argument("--out", type=str, default="summary.png")
    args = ap.parse_args()

    runs = {r: read_metrics(r) for r in args.runs}

    # Reward plot
    plt.figure()
    for r, ms in runs.items():
        xs = [m["step"] for m in ms]
        ys = [m["reward_mean"] for m in ms]
        plt.plot(xs, ys, label=os.path.basename(r.rstrip("/")))
    plt.xlabel("step")
    plt.ylabel("reward_mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)

    # Reference forward cost plot
    out2 = os.path.splitext(args.out)[0] + "_ref_cost.png"
    plt.figure()
    for r, ms in runs.items():
        xs = [m["step"] for m in ms]
        ys = [m["rollout_ref_forwards"] + m["train_ref_forwards"] for m in ms]
        plt.plot(xs, ys, label=os.path.basename(r.rstrip("/")))
    plt.xlabel("step")
    plt.ylabel("ref_forwards_per_step (rollout+train)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out2)

    print("Wrote:", args.out, "and", out2)

if __name__ == "__main__":
    main()
