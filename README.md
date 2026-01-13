# prol-noma-wedge2

A small, runnable repo to test the *principled wedge* we discussed:

1. **Avoiding an explicit reference-model forward** for KL-style regularization by switching to a behavior-policy trust region (Variant **B**).
2. If you do keep an explicit reference model (Variants **A/C**), test **reset semantics**:
   - keep optimizer state
   - hard reset optimizer
   - soft reset optimizer moments

This is intentionally *toy-scale* (CPU friendly) but implements the exact bookkeeping that large GRPO/RLHF systems use:
- rollout logprobs
- multiple update epochs per rollout batch
- optional cached reference logprobs
- reference reset cadence

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .
python -m prol_noma_wedge.run --outdir results/A --variant A --steps 500
python -m prol_noma_wedge.run --outdir results/B --variant B --steps 500
python scripts/plot_results.py --runs results/A results/B --out results/learning.png --out_ref results/ref_cost.png
```

## Variants

- **A**: Explicit KL penalty to a frozen reference model `pi_ref`.
  - `--naive_ref_each_epoch` forces a **ref forward per update epoch** (expensive baseline).
  - default caches `logp_ref` **once per rollout batch** (what real systems do).

- **B**: No reference model. Uses rollout `logp_old` as the anchor (behavior-policy trust region / KL proxy).
  - **Eliminates ref forward entirely** during training.

- **C**: Like A (explicit ref), plus optimizer reset semantics at reference resets:
  - `--optimizer_reset keep|hard|soft`
  - `--soft_reset_factor 0.2` scales Adam moments during a soft reset.

## Tasks

Toy "verifiable reward" sequence tasks:
- `reverse_digits`: prompt is a digit string; answer must be reversed.
- `parity`: answer is `even`/`odd` depending on sum of digits.

Rewards are 1.0 for exact match, else 0.0.

## NOMA integration stub

See `noma/README.md` for how to replace the Python update step with a NOMA script later (file-based handoff).
