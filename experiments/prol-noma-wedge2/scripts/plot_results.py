from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs containing train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_ref", required=True)
    args = ap.parse_args()

    # Learning plot
    plt.figure()
    for r in args.runs:
        rdir = Path(r)
        rows = read_jsonl(rdir / "train.jsonl")
        xs = [row["step"] for row in rows]
        ys = [row["reward_mean"] for row in rows]
        plt.plot(xs, ys, label=rdir.name)
    plt.xlabel("step")
    plt.ylabel("reward_mean")
    plt.legend()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200)
    plt.close()

    # Ref forward cost plot
    plt.figure()
    for r in args.runs:
        rdir = Path(r)
        rows = read_jsonl(rdir / "train.jsonl")
        xs = [row["step"] for row in rows]
        ref = [row["ref_forwards"] for row in rows]
        plt.plot(xs, ref, label=rdir.name)
    plt.xlabel("step")
    plt.ylabel("cumulative ref_forwards")
    plt.legend()
    Path(args.out_ref).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_ref, dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
