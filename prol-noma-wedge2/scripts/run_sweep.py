from __future__ import annotations
import subprocess
from pathlib import Path

def run(cmd):
    print(" ".join(cmd))
    subprocess.check_call(cmd)

def main():
    base_out = Path("results/sweep")
    base_out.mkdir(parents=True, exist_ok=True)

    # Compare A (cached) vs A (naive) vs B (no ref)
    run(["python", "-m", "prol_noma_wedge.run", "--outdir", str(base_out/"A_cached"), "--variant", "A", "--steps", "800"])
    run(["python", "-m", "prol_noma_wedge.run", "--outdir", str(base_out/"A_naive"), "--variant", "A", "--steps", "800", "--naive_ref_each_epoch"])
    run(["python", "-m", "prol_noma_wedge.run", "--outdir", str(base_out/"B"), "--variant", "B", "--steps", "800"])

    # Reset semantics under explicit ref
    run(["python", "-m", "prol_noma_wedge.run", "--outdir", str(base_out/"C_keep"), "--variant", "C", "--steps", "800", "--optimizer_reset", "keep"])
    run(["python", "-m", "prol_noma_wedge.run", "--outdir", str(base_out/"C_hard"), "--variant", "C", "--steps", "800", "--optimizer_reset", "hard"])
    run(["python", "-m", "prol_noma_wedge.run", "--outdir", str(base_out/"C_soft"), "--variant", "C", "--steps", "800", "--optimizer_reset", "soft", "--soft_reset_factor", "0.2"])

    run(["python", "scripts/plot_results.py",
         "--runs",
         str(base_out/"A_cached"),
         str(base_out/"A_naive"),
         str(base_out/"B"),
         str(base_out/"C_keep"),
         str(base_out/"C_hard"),
         str(base_out/"C_soft"),
         "--out", str(base_out/"learning.png"),
         "--out_ref", str(base_out/"ref_cost.png")])

if __name__ == "__main__":
    main()
