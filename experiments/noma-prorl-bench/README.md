# NOMA × ProRL Reset & KL Bench (scoped, runnable)

This repo is a **small, principled benchmark harness** for the ProRL-style idea:
> **Periodic reference policy resets** + **KL regularization** during prolonged RL.

It is designed to let you test two wedges:

1) **"Ref forward cost wedge"**: many implementations accidentally recompute reference logprobs inside inner PPO epochs.
   - `A0` (bad baseline): recompute reference logprobs **each PPO epoch** (`--ref_compute naive_epoch`)
   - `A1` (good baseline): compute reference logprobs **once per rollout batch**, cache and reuse (`--ref_compute cached`)

2) **"State semantics wedge"**: what happens to optimizer state at reset boundaries?
   - paper-style hard reset vs blog-style keep-state vs soft "washout" variants:
     `--opt_reset {keep,hard,soft,asym}`

Optional (NOMA-native) extension:
- **Topology growth** via an adapter rank increase. This makes optimizer-state mapping nontrivial and is where NOMA's `alloc/realloc` semantics become uniquely relevant.
  Enable with `--enable_growth 1`.

---

## What this bench is (and isn't)

✅ It is:
- A minimal PPO-style loop over **verifiable toy sequence tasks** (reverse digits, parity, etc.).
- Explicit accounting for **reference forward passes** and timings.
- Controlled **reference reset cadence** and **optimizer state transforms**.

❌ It is not:
- A full LLM RLHF training system (no distributed rollout engine).
- A claim that "no reference forward is always possible" (it isn't if you need exact ref logprobs).

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you just want to run without installing:
```bash
PYTHONPATH=src python -m noma_prorl_bench.run --help
```

---

## Quickstart

### A0 vs A1: reference logprob recomputation (the speed wedge)
```bash
# BAD baseline: ref logprobs recomputed every PPO epoch
PYTHONPATH=src python -m noma_prorl_bench.run \
  --outdir runs/A0 \
  --variant A \
  --ref_compute naive_epoch \
  --steps 400

# GOOD baseline: ref logprobs cached once per rollout
PYTHONPATH=src python -m noma_prorl_bench.run \
  --outdir runs/A1 \
  --variant A \
  --ref_compute cached \
  --steps 400
```

### Reset semantics: keep vs hard vs soft washout
```bash
# blog-like: keep optimizer state at ref reset
PYTHONPATH=src python -m noma_prorl_bench.run \
  --outdir runs/keep \
  --variant A \
  --ref_compute cached \
  --opt_reset keep \
  --reset_every 200 \
  --steps 800

# paper-like: hard reset optimizer state at ref reset
PYTHONPATH=src python -m noma_prorl_bench.run \
  --outdir runs/hard \
  --variant A \
  --ref_compute cached \
  --opt_reset hard \
  --reset_every 200 \
  --steps 800

# soft washout: decay moments at ref reset
PYTHONPATH=src python -m noma_prorl_bench.run \
  --outdir runs/soft \
  --variant A \
  --ref_compute cached \
  --opt_reset soft \
  --soft_factor 0.2 \
  --reset_every 200 \
  --steps 800
```

### Optional: enable adapter growth (topology change)
```bash
PYTHONPATH=src python -m noma_prorl_bench.run \
  --outdir runs/grow \
  --variant A \
  --ref_compute cached \
  --enable_growth 1 \
  --grow_every 300 \
  --rank_step 4 \
  --steps 1200
```

---

## Plotting

```bash
python scripts/plot_runs.py --runs runs/A0 runs/A1 runs/keep runs/soft --out runs/summary.png
```

This creates:
- `summary.png` (reward + KL + throughput)
- `ref_cost.png` (reference forward calls and timing)

---

## Outputs

Each run writes:
- `config.json`
- `metrics.jsonl` (one JSON per step; easy to parse)
- `events.jsonl` (reset/growth events)
- plots under `outdir/plots/` (optional)

---

## CLI reference

Run:
```bash
PYTHONPATH=src python -m noma_prorl_bench.run --help
```

Key flags:
- `--variant {A,B}`:
  - `A`: explicit reference KL (requires ref logprobs)
  - `B`: no reference model; trust region vs old policy only (no ref forward)
- `--ref_compute {cached,naive_epoch}`:
  - `cached` is what serious implementations do (compute once per rollout)
  - `naive_epoch` is an intentionally wasteful baseline
- `--kl_estimator {simple,k2}`:
  - `simple`: tokenwise (logp_new - logp_ref)
  - `k2`: k2-like estimator that uses logp_ref and logp_old (closer to ProRL v2 blog's flavor)
- `--opt_reset {keep,hard,soft,asym}`:
  - what to do to optimizer state at reference resets
- `--enable_growth 1`:
  - periodically increase adapter rank and migrate optimizer state

---

## Notes for NOMA integration

This repo is written in PyTorch for speed of iteration, but the **interfaces are intentional**:
- cached rollout artifacts are explicit tensors (`logp_old`, `logp_ref`, `tokens`, `mask`)
- reset semantics are implemented as explicit state transforms
- adapter growth includes a *state migration* function for optimizer state

To integrate NOMA, replace:
- the parameter buffers for the adapter (and potentially the whole model)
- the optimizer state transform / migration logic
with NOMA `alloc/realloc`-defined primitives.



### Note on `k2` modes

The NVIDIA ProRL v2 blog writes a `k2` term that (as written) depends on `pi_ref/pi_old`. In a strict PPO implementation,
that quantity is **constant w.r.t. the current parameters** during the inner epochs (because `pi_old` is the snapshot
policy used to sample the data). In this harness:

- `--k2_mode ref_over_old` matches the blog's written ratio and is treated primarily as a **diagnostic** (logged each step).
- `--k2_mode theta_over_ref` makes the `k2` penalty **differentiable** (ratio `pi_theta/pi_ref`) and can be used as the actual KL penalty by setting `--kl_estimator k2`.

If you want a faithful reproduction of the blog's exact training objective, you'll likely also want the blog's
**adaptive beta schedule**; this harness keeps beta fixed to isolate the forward-cost and reset-state effects.

