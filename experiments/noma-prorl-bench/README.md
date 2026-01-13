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



Prior Discussion (not all ideas are well-scoped, but make sure to read for good context):

I’m going to (1) web-check what “dynamic XOR in TTA” refers to in the NOMA context, (2) inspect the NOMA repo to see how realloc/optimizer-state preservation is implemented, and (3) propose concrete, principled optimizer-state transforms (beyond keep/reset) that could matter for ProRL-style resets.


Below is a “give-this-to-coding-agents” document: **what’s actually going on (ProRL v2 + KL resets), what’s expensive vs not, what NOMA’s *real* advantage is (stateful memory + optimizer-state semantics under topology changes), what you should build, and what required web search**.

I’m going to be very explicit about (a) facts grounded in sources, (b) reasonable engineering inferences, and (c) speculative extensions.

---

# NOMA × ProRL(v2) Wedge: what’s real, what’s expensive, and what NOMA can uniquely prove

## 0) Goal of this document (for coding agents)

Coding agents can implement systems, but they typically miss:

* **what ProRL v2 actually resets** and *why*,
* what the **dominant compute cost** is (and isn’t),
* what the **KL estimator requires** (data dependencies),
* how **existing RLHF stacks** structure “actor/ref/old logprobs” and where caching naturally lives,
* what NOMA’s **unique capability** is (it’s *not* “a faster copy”; it’s “explicit, mechanically-defined state transforms under realloc/topology changes”).

This doc gives you:

1. the conceptual wedge,
2. the “what to implement” scope,
3. the web-researched facts you should forward to agents,
4. the repository + code context from the last implementation (the zip you have).

---

## 1) What the wedge is (one sentence)

**Show that treating “reference/old-policy statistics + optimizer state” as first-class, explicitly-managed buffers (à la NOMA) eliminates avoidable reference recomputation and enables *controlled state resets/retention* that PyTorch makes brittle—especially once you add topology changes (grow/shrink) or selective optimizer washout.**

That splits naturally into two tiers:

### Tier A (minimal, “one-shot” plausible)

**Make the reference forward “not multiply” with PPO epochs / minibatches** by caching the required reference terms once per rollout and reusing them correctly.

This is a wedge because many systems *accidentally* recompute reference quantities multiple times (or re-materialize buffers wastefully), and you can show a clear speed/step-time delta on a small benchmark.

### Tier B (the “NOMA-unique” wedge that your message is pointing at)

**Implement “stateful reset semantics”**: not just keep/reset optimizer state, but **selectively transform it** (soft reset / washout / diagonalized second-order surrogates / growth-aware mapping) at “reference reset” boundaries or “capacity growth” boundaries.

This is where NOMA actually has teeth: NOMA explicitly frames dynamic topology and asks “what happens to optimizer state when parameters are grown or remapped?” and highlights that this is painful/error-prone in standard frameworks. ([GitHub][1])

---

## 2) What ProRL v2 is doing with KL regularization + resets (facts)

From NVIDIA’s ProRL v2 writeup:

* They use **KL-regularized trust regions** with **periodic reference resets**. ([NVIDIA Developer][2])
* The KL regularization they describe uses a **k₂ estimator** involving a ratio between the **reference policy** and the **old policy** log-probabilities (they explicitly define (x = \exp(\mathrm{clamp}(\log(\pi_\text{ref}/\pi_{\text{old}}))))). ([NVIDIA Developer][2])
* **Reset cadence:** “Every 200–500 RL steps (or upon KL spikes/stalled validation), the reference policy is reset to the current policy; optimizer state is not cleared.” ([NVIDIA Developer][2])

So in ProRL v2, the reset event is **not** “reset the optimizer”; it’s primarily “update the anchor (reference policy) to a newer checkpoint” to prevent being constrained by an outdated anchor. ([NVIDIA Developer][2])

### Important: ProRL (v1 paper) differs

The original ProRL paper (arXiv html) describes a reset that **reinitializes optimizer states** when they reset to the best policy checkpoint. ([arXiv][3])
So when you talk to agents: **be precise whether you’re matching ProRL v1 (optimizer reset) or ProRL v2 (optimizer not cleared).**

This matters because it changes what your “optimizer state wedge” is:

* For **v2**, the wedge is *not* “they reset optimizer and we don’t” (they already don’t).
* For **v1**, the wedge could be “full reset is too blunt; do selective washout instead.”

---

## 3) What is being reset? (disambiguation that agents must internalize)

There are *at least* 4 distinct “states” people conflate:

### (S1) Reference policy parameters (π_ref)

Reset = **set π_ref ← π_current(best)** periodically.
This is what ProRL v2 explicitly describes.

### (S2) Old policy statistics (π_old logprobs)

In PPO-style updates, you usually keep π_old fixed across the K epochs/minibatches for a rollout batch (or across a “mini-iteration”). ProRL v2’s KL ratio explicitly uses π_old vs π_ref.
So you need **logp_old** available at update time.

### (S3) Optimizer state (Adam moments / momentum / second-moment)

This is the thing you’re pointing at with “dynamic XOR / preserve optimizer state.”

* ProRL v2 says optimizer state is **not cleared** at reference reset.
* ProRL v1 describes reinitializing optimizer state at reset.

### (S4) Cached auxiliary tensors (reference logprobs, values, advantages, etc.)

These are not “optimizer state,” but they are the *main mechanism* to avoid redundant computation: if you compute reference logprobs once per rollout, you shouldn’t recompute them each epoch.

If agents mix up (S1)-(S4), they will design the wrong benchmark.

---

## 4) What’s expensive vs what isn’t (the cost model agents should use)

### 4.1 The expensive part is the *reference forward*, not the reset event

Almost every RLHF PPO-style recipe computes token logprobs under:

1. the trainable policy, and
2. a reference policy,
   then uses the KL between them as regularization. TRL’s PPO trainer docs spell out that optimization computes logprobs with the trained model and a reference model to form a KL divergence signal.
   The RLHF book similarly describes a frozen initial model computing log-probabilities on the same text to compute a KL penalty.

**Compute implication:** a reference model forward pass is “one extra forward pass” over the sequence(s) you train on. That’s not free.

### 4.2 The reset event itself (copying / swapping π_ref) is usually cheap amortized

Resetting π_ref every 200–500 steps means you do an occasional parameter copy/update; amortized over hundreds of steps, it’s small relative to constant per-step reference forward.

If someone tells you “the reset is expensive,” they’re usually confusing:

* the **ongoing cost** (reference forward each rollout or minibatch) vs
* the **occasional event** (update π_ref weights).

### 4.3 Where “NOMA might save compute” is *not* in copying weights

If your wedge is “NOMA makes the reset copy faster,” that’s weak and probably unconvincing.
The convincing wedge is:

* **avoid redundant recomputation** (reference forward multiplied by epochs),
* **stabilize memory/buffer management** so you don’t reallocate/reshard unpredictably,
* **enable explicit state transforms** when shapes/topology change.

---

## 5) Why your “dynamic XOR / TTA” point matters (and what it really implies)

NOMA’s own README frames dynamic topology as:

* make parameters explicit buffers (`alloc/realloc/free`),
* and **mechanically define what happens to optimizer state** when topology changes.

They explicitly highlight that **standard frameworks don’t handle “evolving topology” cleanly**, and that “existing optimizer state is preserved… leading to faster convergence… than reinitialization.”

That’s your “dynamic XOR” intuition: if the model grows, **keeping optimizer moments for the *old* parameters** makes the system continue learning instead of “forgetting how to step.”

### The key translation to ProRL:

ProRL’s reference reset (π_ref update) does *not* change topology. So NOMA’s “topology advantage” doesn’t automatically apply.

To make NOMA’s optimizer-state superpower relevant, you need at least one of:

1. **selective optimizer state transforms** at reset boundaries (even without topology change), or
2. **capacity/topology adaptation during prolonged RL** (e.g., grow LoRA rank / add features / widen layers) triggered by stagnation / KL spikes / reset events.

That’s the principled bridge between “dynamic XOR keeps learning” and “ProRL resets.”

---

## 6) What existing implementations already do (and where your wedge actually is)

### 6.1 Many libraries already cache “old logprobs”

PPO requires old policy logprobs; it’s standard to compute them at rollout time and reuse.

### 6.2 Reference logprobs: libraries vary

Some compute reference logprobs once per rollout batch and reuse.
Some accidentally recompute per epoch or per minibatch (especially in naive baselines or when mixing distributed roles).

**Your earlier repo (the zip you have) explicitly includes both modes**:

* `naive_ref_each_epoch=True` = intentionally bad baseline (recompute reference logprobs during every PPO epoch/minibatch).
* cached reference logprobs computed during rollout = better baseline.

This is a good benchmark pattern because it creates a measurable gap on small models.

### 6.3 ProRL v2’s KL ratio uses π_ref / π_old

This is important: the k₂ estimator they write uses π_ref and π_old, not necessarily π_ref and π_current directly.
So your cache should (at minimum) keep:

* `logp_old` (behavior policy at rollout),
* `logp_ref` (reference anchor at rollout time, or computed once per rollout),
* and then the current policy logp is recomputed during training (that’s necessary).

---

## 7) The “how can NOMA avoid the extra reference forward?” question (principled answers)

You asked this earlier and it’s easy to answer sloppily, so here’s the precise breakdown.

### Option 1 (standard + correct): **compute reference logprobs once per rollout, reuse across PPO epochs**

This does **not eliminate** the reference forward completely, but it ensures it happens at most once per rollout batch.

This is the minimal wedge and is what your current repo demonstrates.

**What NOMA contributes:** treat the cached `logp_ref` buffer as a first-class alloc/realloc-managed artifact rather than an ad-hoc tensor that gets rematerialized or copied across devices.

### Option 2 (stronger): **move reference evaluation to rollout workers and never do it in the optimizer loop**

In distributed RLHF, you often have rollout engines and training engines. If the rollout stage can compute `logp_ref` (teacher forcing on the generated sequences), then the training loop never runs the reference model at all. (It just consumes cached `logp_ref`.)

This can be meaningful when your training engine is already saturated and reference forward steals capacity.

**But** it still requires a reference forward somewhere. It’s just relocated.

### Option 3 (actually “avoid reference forward”): **replace π_ref with a surrogate that doesn’t require a model forward**

This is where you must be honest: you’re approximating.

Three principled-ish surrogates:

1. **Behavior trust region**: penalize KL(π_current || π_old) instead of KL to π_ref. Your repo’s Variant B does this. It eliminates the reference model entirely, but changes the algorithm.
2. **Periodic anchor logprobs only**: at reset time, compute and store a “reference logprob table” for a fixed evaluation set and use it for a partial penalty. This is weak unless you carefully justify why it regularizes the relevant distribution.
3. **Local quadratic / Fisher surrogate**: approximate KL to reference using a curvature estimate around π_ref, so you don’t need π_ref forward each time. This is researchy and needs careful math + validation.

If you want a wedge that’s “one-shot to code,” Option 1/2 is the realistic path.

---

## 8) Optimizer-state “washout” (what you’re gesturing at, made concrete)

You described:

* “preserve optimizer state and not reset it”
* “washout old directions”
* “second-order diagonalized on the fly”
* “log learning rate schedules work on this”

Here’s a principled, implementable translation.

### 8.1 Define the event: when do we apply washout?

Candidate triggers:

* every reference reset (every 200–500 steps),
* or when KL spikes / stalled validation (also mentioned by NVIDIA),
* or when reward plateaus.

### 8.2 Define the transform family on optimizer state

For Adam-like optimizers with `m` (first moment) and `v` (second moment):

**Hard reset**: `m ← 0, v ← 0`
**Keep**: no change
**Soft reset (scalar)**: `m ← α m, v ← α v` with α ∈ (0,1)
**Asymmetric washout** (often more sensible):

* `m ← α m` but keep `v` (or vice versa), because `v` encodes curvature/scale and `m` encodes direction/momentum.

**Diagonal-second-order pivot**: interpret `1/sqrt(v)` as diagonal preconditioner; you can keep `v` but reduce `m` so you keep stable step scaling but forget stale direction.

These are easy to implement and easy to ablate.

### 8.3 Why NOMA matters here (vs “PyTorch can do it too”)

If you **never change topology**, PyTorch can absolutely apply these transforms. So NOMA doesn’t win by default.

NOMA wins when either:

* you want to formalize “reset semantics” as part of a general compute graph / compiled pipeline, or
* you introduce **topology change** (next section), where mapping optimizer state across a structural change becomes painful.

So: implement washout anyway (it’s useful), but don’t oversell it as NOMA-exclusive unless you add topology change.

---

## 9) The real NOMA-native experiment (if you want something genuinely “Torch can’t do cleanly”)

### 9.1 Add a **tiny topology adaptation** to prolonged RL

Trigger: at the same cadence as reference resets or stagnation.

Examples that are *still scoped*:

* **Grow a low-rank adapter** (increase LoRA rank) in one module.
* **Widen one MLP layer** by adding neurons.
* **Add a small “policy head expansion”** (extra hidden units) on a toy model.

### 9.2 The core requirement: preserve optimizer state for the *old* slice

When you grow parameters, you want:

* existing weights keep their optimizer moments,
* new weights get fresh moments (zero/init),
* any remapping is explicit.

This is precisely the “dynamic XOR converges faster because optimizer state is preserved” argument NOMA makes.

### 9.3 Why this is a strong wedge

Because in plain PyTorch, doing growth “correctly” requires:

* constructing new parameters,
* copying old weight values into the new tensor,
* and correctly migrating optimizer state (m, v) keyed by parameter objects (which often breaks silently).

NOMA’s whole pitch is that realloc is a primitive and optimizer-state semantics are defined.

So if you want a wedge that makes people say “okay, that’s actually different,” this is it.

---

## 10) What you already have in the “prol-noma-wedge2” repo (previous design decisions)

This is what your agents should know about the last implementation so they don’t reinvent or misunderstand it.

### 10.1 What it benchmarks

A small PPO-style loop with:

* a **policy model** (tiny GRU LM),
* an optional **reference model**,
* a toy task (“reverse_digits” etc),
* counters tracking:

  * policy forwards during rollout + training,
  * reference forwards.

### 10.2 The intended “reference forward multiplier” bug

There is a deliberate switch:

* `naive_ref_each_epoch=True` recomputes reference logprobs during each PPO epoch/minibatch (expensive baseline).
* otherwise it computes `ref_logp` once per rollout and reuses it.

This isolates the “unnecessary recomputation” wedge.

### 10.3 It already includes optimizer reset semantics

`optimizer_reset ∈ {keep, hard, soft}` where soft multiplies Adam moments by a factor.

That directly maps to the “washout” family above.

### 10.4 NOMA integration was intentionally stubbed

Your `INTEGRATION_TO_NOMA.md` basically said:

* “the wedge is explicit cached buffers + reset semantics,”
* “for real NOMA, implement cache allocation + reuse through NOMA alloc/realloc or similar abstractions.”

So your agents should treat the repo as a harness and slot in NOMA primitives, not rewrite the algorithm from scratch.

---

## 11) Search-required findings you should forward verbatim-ish to coding agents (so they don’t need search)

This is the part you explicitly asked for: **everything that required search and is important context**.

### 11.1 ProRL v2 reset behavior (NVIDIA blog)

Key facts:

* ProRL v2 uses **KL-regularized trust regions** with **periodic reference resets**.
* They describe a **k₂ estimator** that depends on (\pi_\text{ref}) and (\pi_{\text{old}}) (ratio inside a clamp).
* Reset cadence: **every 200–500 RL steps** or on **KL spikes / stalled validation**; **reference reset to current policy**; **optimizer state not cleared**.

### 11.2 ProRL v1 (paper) differs on optimizer reset

The ProRL paper describes resetting to the best checkpoint and **reinitializing optimizer state** at reset.

Agents must not mix these up.

### 11.3 NOMA’s claim about optimizer state + dynamic topology

NOMA frames dynamic topology as alloc/realloc/free for parameter buffers and explicitly calls out the need to define optimizer-state behavior; it states that preserving optimizer state across realloc leads to faster reconvergence in their XOR growth demo.

### 11.4 Reference model forward is “standard RLHF cost”

TRL’s PPO trainer describes that optimization computes logprobs with the trained model **and a reference model** and uses KL divergence to prevent drift.
RLHF book similarly describes a frozen initial model computing log-probabilities on the same text to calculate a KL penalty.

These are the conceptual basis for “reference forward is a real cost center.”

### 11.5 (Optional background) KL estimator discussion

There’s active work on KL-regularized policy gradient design and token-level KL penalties (k₂-style estimators show up in modern RLVR stacks). One recent OpenReview paper explicitly discusses KL-regularized policy gradient design and mentions k₂/token-level KL penalties in the LLM RL context.

(Agents don’t need to read it to code the harness, but it supports “this is real, not made up.”)

---

## 12) Concrete “what to build next” (so agents can execute)

### Phase 1 (fast, scoped): reproduce the wedge cleanly

Implement (or keep) these variants on the small harness:

**Variant A0 (bad baseline)**

* recompute reference logprobs every PPO epoch/minibatch (`naive_ref_each_epoch=True`)

**Variant A1 (good baseline / correct caching)**

* compute reference logprobs once per rollout batch; store; reuse.

**Variant A2 (buffer-stability / NOMA-ish)**

* implement a persistent “reference logprob ring buffer” abstraction:

  * preallocate max size,
  * reuse without realloc,
  * support slicing/batching without copies.

Even if this is implemented in PyTorch first, it’s the object that maps to NOMA alloc/realloc.

**Metrics**:

* wall clock per step,
* counts of reference forwards,
* peak memory if measurable,
* reward curves / KL curves.

### Phase 2 (the optimizer-state part you actually care about)

On top of A1/A2, add:

**Reset semantics ablation** (at reference reset boundaries):

* keep optimizer state (ProRL v2 style)
* hard reset (ProRL v1 style)
* soft washout (α on m/v)
* asymmetric washout (reset m, keep v)

**Key measurement**: stability after resets (KL spikes, reward collapse, recovery speed).

### Phase 3 (the real NOMA “dynamic XOR” bridge)

Add one controlled topology change:

* e.g., widen a tiny layer by +K units every N resets
* preserve optimizer state for existing slice, init new slice

Then compare:

* naive PyTorch “new params” (state migration mistakes happen easily),
* correct PyTorch state migration (painful but doable),
* NOMA-style realloc semantics (the conceptual target).

This is the only phase where you can honestly claim “NOMA enables something Torch makes brittle,” aligned with NOMA’s own story.

---

## 13) If the research changes the experiment (your explicit request)

Yes: **the NVIDIA ProRL v2 writeup explicitly says optimizer state is not cleared at reference resets**.
So if your earlier mental model was “they reset reference *and* optimizer,” that’s **not** ProRL v2.

**Implication:**

* The strongest ProRL v2-aligned wedge is **reference-stat caching/bufferization** and avoiding redundant reference computation.
* The strongest optimizer-state wedge is either:

  * aligning to **ProRL v1** (which resets optimizer), or
  * adding **topology changes** so optimizer-state semantics actually matter in a way that’s not trivial in PyTorch.

---

## 14) The “insanely grounded” narrative you can tell in a writeup / PR

If you want something that reads like a serious experiment and not vibes:

1. ProRL v2 uses KL trust regions and periodically resets π_ref (every 200–500 steps) without clearing optimizer state.
2. KL regularization in RLHF commonly requires reference logprobs, which implies extra reference forward compute.
3. Many implementations accidentally amplify that cost by recomputing reference quantities inside inner optimization loops. (Your harness demonstrates this with A0 vs A1.)
4. NOMA’s alloc/realloc framing makes “policy statistics as persistent buffers” and “optimizer-state semantics under evolution” a first-class design axis; it explicitly calls out preserving optimizer state under realloc as beneficial (XOR demo).
5. Therefore, we benchmark:

   * reference-forward recomputation elimination (A0 → A1 → A2),
   * optimizer-state reset semantics (keep/hard/soft),
   * and (optionally) topology growth with optimizer-state preservation (the true NOMA-native case).

That’s principled, testable, and doesn’t rely on magical claims.

---

If you want, paste whatever your agents produce (timings + curves + any surprising failure mode). I’ll help you diagnose whether the effect is real, whether the benchmark is fair, and what the next tight ablation should be.

[1]: https://github.com/pierridotite/NOMA "GitHub - pierridotite/NOMA: A systems programming language where automatic differentiation is a compiler pass and model parameters are explicit, growable memory."
[2]: https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/ "Scaling LLM Reinforcement Learning with Prolonged Training Using ProRL v2 | NVIDIA Technical Blog"
[3]: https://arxiv.org/html/2505.24864v1?utm_source=chatgpt.com "ProRL: Prolonged Reinforcement Learning Expands ..."


Below is a “handoff doc” you can paste straight to coding agents. I’m going to (1) nail down what the **reference reset** actually is (and what exactly gets “reset”), (2) break down **where the compute/memory cost really comes from** in real RLHF/GRPO stacks, (3) summarize **what existing systems do**, (4) map that onto **what NOMA can plausibly help with**, and (5) scope the **most principled experiments** (toy → real) that can produce a clean “cool result.”

---

## 0) The actual target: what problem are we solving?

You’re reacting to a standard pain point in KL-regularized RLHF / GRPO / PPO-style post-training:

* You want the **stability benefits of KL regularization to a reference policy** (prevents drift / reward hacking / collapse).
* But during long/prolonged training, **the KL term can dominate** and you stop making meaningful policy progress.
* A pragmatic trick is **reference policy reset**: periodically move the reference forward so it’s not “too far behind,” so KL stops acting like a hard leash.

That’s the “reset every few hundred steps” point.

### What ProRL (paper) claims the reset is

ProRL explicitly motivates resets by saying the **KL term increasingly dominates** the loss and policy updates diminish; then proposes **reference policy reset**: “hard-reset the reference policy to a more recent snapshot of the online policy” **and reinitialize optimizer states**. ([arXiv][1])

### What NVIDIA’s ProRL v2 blog claims the reset is

NVIDIA’s ProRL v2 blog says: every ~200–500 RL steps (or when KL spikes / validation stalls), **reset the reference policy to the current policy** and **optimizer state is not cleared**. ([NVIDIA Developer][2])

**Important discrepancy (principled note):**

* Paper: reset reference **and** reinit optimizer state.
* Blog: reset reference, **do not** clear optimizer.
  These are meaningfully different “reset semantics.” You should treat them as **two real variants** (and test both).

---

## 1) What is being “reset” in practice?

There are *two different* things that people loosely call “reset,” and mixing them causes confusion:

### (A) Resetting the **reference policy**

This is:

* `π_ref ← snapshot(π_θ)` (often: copy weights from actor → ref model, or pointer swap)
* Purpose: make KL penalty measure divergence to a *recent* anchor instead of the original SFT anchor.

This is the reset both the ProRL paper and NVIDIA blog talk about. ([arXiv][3])

### (B) Resetting the **optimizer state**

This is separate:

* “Hard reset” = reinitialize Adam moments to zero (and sometimes reset LR schedulers).
* “Soft reset” = decay moments (multiply by factor) instead of zeroing.

ProRL paper explicitly says optimizer states are reinitialized at reset. ([arXiv][1])
NVIDIA blog says optimizer state is not cleared. ([NVIDIA Developer][2])

### What you should tell coding agents (key clarity)

When someone says “KL reset,” ask: **which** reset?

* Reference reset? (`π_ref` update)
* Optimizer reset? (Adam moments / scheduler)
* Or both?

Because the cost + behavior changes depend on which one you do.

---

## 2) Where the *real* cost is: reference forward vs reset ops

You asked: “Is it a big operation like state, or a small cache thing?”

### Cost bucket #1: The **reference-model logprob computation** (the “extra forward”)

KL regularization needs tokenwise log-prob under **both**:

* current policy `log π_θ(a_t|s_t)`
* reference policy `log π_ref(a_t|s_t)`

Unless you do something clever, `log π_ref` requires a **separate forward pass** through a second model (the ref copy).

Most serious RL stacks expose explicit knobs for the reference logprob forward (micro-batching, offload, etc.), which is a dead giveaway that this forward is a major cost center. For example, VERL’s config docs explicitly describe:

* enabling a separate reference model when KL is used
* separate `ref.log_prob_micro_batch_size(_per_gpu)` for computing `ref_log_prob`
* recommendations to offload ref for >7B models ([Verl][4])

**Rule-of-thumb cost impact (grounded, but order-of-magnitude):**

* Training step for actor is roughly **forward + backward**.
* Backward is typically ~2× forward (varies).
* So actor update ~3 “forward-equivalents.”
* Adding a reference forward adds ~1 forward-equivalent.
* That’s ~**+33%** extra compute on the training side, *if* you were already doing the actor forward for logprobs.
* But in RLHF/GRPO pipelines, rollout generation can dominate, and reference scoring can add another big chunk depending on where it’s computed.

**This is the piece you were calling “the extra reference forward.”**
This is *not* a “small cache thing.” It’s real compute.

### Cost bucket #2: The **reference reset operation** (copying weights / swapping)

If reference is a full model copy, resetting reference means either:

* copying weights actor → ref (O(#params)), or
* swapping pointers (if you keep versions), or
* updating which checkpoint ref loads from.

This can be cheap-ish on a single GPU (memcpy) but can be painful in multi-GPU sharded setups (needs collectives / reshards).

### Cost bucket #3: The **optimizer-state reset**

If you do the ProRL-paper style optimizer reinit, that is:

* allocate/zero Adam moments (often fp32)
* potentially reinit schedulers

For big models, optimizer state can be *multiple times* model weight size. Reinitializing it is not a “tiny cache;” it’s a lot of memory traffic.

### Cost bucket #4 (the sleeper): **refit / reshard / weight transfer** overhead

In multi-component RL systems (separate rollout engine vs training engine), a major pain is “refit” time: transferring weights between sharding layouts / engines.

A concrete data point: a NeMo RL discussion reports “refit time reduced from 693s to 47s” with changes, and calls out how serious refit overhead is. ([NVIDIA Docs][5])

This matters because reference resets (and frequent snapshotting) can force more frequent weight movement or re-configuration depending on architecture.

---

## 3) What’s happening mathematically when KL dominates?

Most GRPO/PPO-ish variants effectively optimize something like:

* maximize reward (advantage-weighted logprob)
* plus/minus a KL penalty that keeps policy near reference

NeMo RL’s GRPO math guide gives an explicit **per-token KL approximation** commonly used in practice:
[
D_{KL}(\pi_\theta||\pi_{ref}) \approx \frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1
]
and includes the KL term in the loss. ([NVIDIA Docs][6])

**Why KL dominates over time (intuitive, but precise enough):**

* As the policy improves reward, it tends to move away from the initial reference.
* KL penalty grows with that divergence.
* Eventually the gradient contribution from “don’t move further” overwhelms the gradient from “get more reward,” and updates shrink.

**Reference reset is literally a hack to make KL small again** by redefining what “close” means: instead of “close to the original SFT,” it’s “close to what I was doing recently.”

---

## 4) What existing implementations do (patterns you can assume)

Even without reading every repo, the broad architecture is consistent across TRL / VERL / NeMo RL / OpenRLHF-style systems:

### Pattern 1: Reference logprobs are *usually computed once per rollout batch and cached*

Because if you do multiple epochs of policy updates per batch, recomputing ref logprobs per epoch is wasteful.

In TRL-style trainers, there are explicit flags like disabling dropout “useful for training with a reference model” (because nondeterminism breaks cached comparisons). ([Hugging Face][7])

So the “best practice” is:

* generate rollouts
* compute `logp_policy_old` and `logp_ref` for the rollout tokens
* store them in the batch
* reuse across update epochs

### Pattern 2: Ref model is often sharded/offloaded

VERL docs explicitly recommend offload for ref for >7B. ([Verl][4])
That implies “reference forward” is expensive enough to warrant special memory plumbing.

### Pattern 3: Reference reset cadence is heuristic and ops-driven

NVIDIA’s ProRL v2 blog: reset every 200–500 RL steps, or when KL spikes / validation stalls. ([NVIDIA Developer][2])
That’s a systems heuristic, not a clean theorem.

---

## 5) Where NOMA actually fits (what is plausible vs not)

### What NOMA *is* (relevant capability)

NOMA is explicitly positioned to support:

* “growable parameter buffers”
* **preserving optimizer state** across buffer reallocations / topology changes
* avoiding “catastrophic instability” when parameterization changes mid-training ([NVIDIA Docs][6])

So NOMA is naturally about **state semantics**: what happens to optimizer moments, and how you carry them through nontrivial changes.

### What NOMA probably cannot do (be honest, stay principled)

If you are full-finetuning dense transformer weights and need `log π_ref` under a different set of weights than `π_θ`, you cannot magically get exact ref logprobs without doing equivalent work somewhere. The KL needs two distributions.

So NOMA will not “eliminate” reference forward *in the general case*.

### Two places NOMA *can* create a real wedge

#### Wedge A: **Reset semantics** (optimizer state mapping) as a first-class experiment

Because ProRL paper explicitly says resets include optimizer reinit, and NVIDIA blog says they do not. ([arXiv][3])
That is screaming for an experiment:

* Variant: ref reset + **hard optimizer reset**
* Variant: ref reset + **keep optimizer state**
* Variant: ref reset + **soft-reset moments** (scale moments, keep directionality)

This is exactly the kind of “state semantics” NOMA is about. ([NVIDIA Docs][6])

If you show:

* “keeping optimizer state stabilizes resets”
  or
* “soft reset dominates both”
  that’s a clean, publishable-ish systems result, and it’s *aligned with NOMA’s stated mission*.

#### Wedge B: Avoiding the explicit reference forward by changing the anchor (algorithmic swap)

This is what the toy repo you have (`prol-noma-wedge2.zip`) already encodes:

* **Variant A:** explicit KL to frozen reference (costly reference scoring).
* **Variant B:** no reference model; use behavior-policy trust region anchored on `logp_old` (collected during rollout) — eliminates `π_ref` forward entirely.

This is an *algorithmic* wedge: it’s not the same objective as reference KL, but it’s a principled approximation in the regime where references are frequently reset forward (because then ref ≈ recent policy anyway).

This is the “avoid extra reference forward” path, but you must be disciplined in framing:

* You are not computing *the same KL*.
* You are substituting a *moving anchor* (behavior policy / old policy), which is a different regularizer.

---

## 6) Prior art you need to know (so you don’t reinvent something obvious)

### (1) ProRL (reference reset) — the core thing you’re targeting

* ProRL explicitly: KL dominates → diminishing updates → **reference policy reset** and **optimizer reinit**. ([arXiv][1])

### (2) NVIDIA ProRL v2 blog — operational guidance + different reset semantics

* Reset every 200–500 RL steps
* Reset reference to current policy
* Optimizer state **not cleared** ([NVIDIA Developer][2])

### (3) Elastic Reset (different reset concept, no explicit KL)

Elastic Reset is a separate “reset family” idea:

* periodically reset the *online model* to an EMA of itself, and reset EMA to initial
* claims better reward vs drift tradeoff without the same KL objective ([arXiv][8])

It’s not the same as reference KL reset, but it’s relevant because it’s another “reset solves drift/overoptimization” story. If you claim novelty, you must distinguish clearly.

### (4) Systems reality: weight transfer/refit can dominate

The NeMo RL refit-time discussion is an existence proof that “resetting things” can turn into giant multi-minute overhead if it triggers weight movement. ([NVIDIA Docs][5])

---

## 7) What I did *not* find on X (important to say plainly)

You asked to search X/Twitter for notes. In practice, X is inconsistent to crawl/search reliably (often blocked/partial). I did not find a clean “canonical thread” that adds more than the ProRL paper + NVIDIA blog + RLHF systems docs above. So: don’t assume “X has the secret implementation detail.” Treat the sources above as the grounded baseline.

---

## 8) The experiment plan that is maximally scoped + maximally meaningful

You want something a coding agent can implement in 1–2 shots and that produces a “cool” plot.

### Stage 0: Toy proof (already in your zip repo)

Repo implements:

* toy verifiable-reward tasks (`reverse_digits`, `parity`)
* rollout batching
* multiple update epochs per rollout
* reference logprob caching vs naive-per-epoch
* reference reset cadence
* optimizer reset semantics keep/hard/soft

**Why this matters:** it isolates *bookkeeping and reset semantics* without distributed complexity.

**What to measure (minimum):**

* reward vs step
* KL estimate vs step (if using explicit ref)
* “policy update magnitude” proxy (e.g., mean |Δlogp| on batch)
* wall-clock per step split into:

  * rollout time
  * training time
  * reference scoring time (explicitly)

### Stage 1: “Real-ish” single-GPU GRPO/TRL run (still scoped)

Use a small-ish open model (0.5B–3B) and a verifiable task set (math/format checking). Keep it single GPU to avoid refit/reshard confounds.

**Variants (these map cleanly onto ProRL confusion):**

1. Ref reset + hard optimizer reset (paper-like)
2. Ref reset + keep optimizer state (blog-like)
3. Ref reset + soft optimizer reset (NOMA-meaningful)
4. No ref reset (control)
5. (Optional) behavior-policy anchor (no ref forward) — clearly labeled as algorithmic change

**Key metric:** “continued improvement after KL would otherwise clamp.” That is literally ProRL’s motivation.

### Stage 2: The “avoid extra reference forward” demo that’s actually defensible

If you want to claim you avoided the ref forward, you must show:

* Baseline system does explicit reference scoring forward.
* Your variant removes that forward and keeps training stable.

The toy repo’s Variant B does that by anchoring to rollout `logp_old`.

**How to frame it so it’s not bullshit:**

* “We replace the fixed reference KL with a behavior-policy trust region regularizer, motivated by the fact that ProRL-style reference resets move the reference forward frequently anyway.”
* Then show: comparable reward curve, improved throughput.

---

## 9) What coding agents usually get wrong here (preempt them)

### Mistake 1: recomputing reference logprobs every update epoch

This explodes cost and is not representative. You need “compute once per rollout batch, cache, reuse.”

### Mistake 2: conflating “reference reset” with “optimizer reset”

They must be separate toggles.

### Mistake 3: ignoring dropout nondeterminism

If dropout differs between ref scoring and policy scoring, KL estimates become noisy. TRL explicitly calls out disabling dropout as useful when training with a reference model.

### Mistake 4: making claims of “same objective” when switching anchors

If you remove the reference model forward by anchoring to `logp_old`, it’s a different regularizer. The benefit can be real, but the claim must be correct.

### Mistake 5: benchmarking “reset cost” without including real systems overhead

On a single GPU, weight copy looks cheap. In real sharded systems, resets can interact with refit/weight movement and become huge. Use the NeMo RL refit discussion as the “systems caution” story.

---

## 10) What changed in the experiment plan after research? (big deltas)

### Delta #1: You **must** treat optimizer reset semantics as an explicit axis

Because paper vs blog disagree. That’s not a footnote; it’s an experiment axis.

### Delta #2: “Avoid extra reference forward” is *not* a guaranteed NOMA win unless you change the algorithm

If you want to truly eliminate ref forward while keeping *exact* reference KL, that’s generally not possible without approximations or architectural tricks. The clean path is:

* either cache ref logprobs (standard)
* or remove the explicit reference KL objective (Variant B-style anchor swap)

So your claim should be: **NOMA helps formalize and preserve state across reset events**, and/or enables a **safe anchor swap** experiment that trades ref forward for throughput.

---

## 11) SEARCH-ENABLED PACKET (send this to agents verbatim)

These are the web-derived facts + references that coding agents wouldn’t have without search:

1. **ProRL paper**: KL dominates → diminishing updates → reference policy reset (hard reset ref to recent snapshot) **and reinitialize optimizer states**. ([arXiv][1])

2. **NVIDIA ProRL v2 blog**: reset reference every 200–500 RL steps; ref reset to current policy; **optimizer state is not cleared**.

3. **NeMo RL GRPO math**: explicit KL approximation and loss structure used in practice (tokenwise KL approx formula). ([NVIDIA Docs][6])

4. **VERL docs**: reference model is an explicit component for KL; separate config knobs for `ref_log_prob` microbatching and recommendations for offload.

5. **NOMA README**: NOMA’s stated purpose includes growable parameter buffers + preserving optimizer state across realloc/topology changes to avoid catastrophic instability. ([NVIDIA Docs][6])

6. **NeMo RL systems note**: refit/weight-transfer overhead can be massive (example numbers like 693s→47s), relevant caution for any scheme that increases weight movement frequency.

7. **Elastic Reset**: another “reset family” approach (EMA-based) that mitigates drift without explicit KL objective; relevant prior art to cite/contrast.

### Links (allowed here in code block so agents can click)

```text
ProRL (paper, arXiv HTML): https://arxiv.org/html/2505.24864v1
NVIDIA ProRL v2 blog: https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/
NeMo RL GRPO math guide: https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/grpo.html
VERL config (reference model + ref_log_prob batching): https://verl.readthedocs.io/en/latest/examples/config.html
NeMo RL refit-time discussion: https://github.com/NVIDIA/NeMo-RL/discussions/60
NOMA repo README: https://github.com/pierridotite/NOMA
Elastic Reset (arXiv): https://arxiv.org/html/2312.07551v1
```

---

## 12) Concrete instructions to agents using your existing toy repo (so they don’t thrash)

Your zip repo (`prol-noma-wedge2.zip`) already encodes the correct experimental axes:

* Variant A: explicit reference KL
* Variant B: behavior-policy anchor (no ref forward)
* Variant C: reset semantics (keep/hard/soft moments)
* Flags to force naive ref forward each epoch (expensive baseline) vs cached once per rollout (realistic)

If they want to extend it:

* add a “paper-like reset semantics” preset (hard reset optimizer on ref reset)
* add a “blog-like reset semantics” preset (keep optimizer on ref reset)
* add a “NOMA semantics” preset (soft reset moments)

And instrument:

* time spent in ref scoring
* time spent in training forward/backward
* total step wallclock
* reward and KL curves around reset boundaries

---

If you come back with early curves (even toy), I can tell you quickly whether you’re seeing the *actual* ProRL phenomenon (KL clamp → reset unlocks progress) or just noise from toy rewards.

[1]: https://arxiv.org/html/2505.24864v1 "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models"
[2]: https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/?utm_source=chatgpt.com "Scaling LLM Reinforcement Learning with Prolonged ..."
[3]: https://arxiv.org/html/2505.24864v1?utm_source=chatgpt.com "ProRL: Prolonged Reinforcement Learning Expands ..."
[4]: https://verl.readthedocs.io/en/latest/examples/config.html?utm_source=chatgpt.com "Config Explanation - verl documentation - Read the Docs"
[5]: https://docs.nvidia.com/nemo/rl/0.3.0/apidocs/nemo_rl/nemo_rl.algorithms.grpo.html?utm_source=chatgpt.com "nemo_rl.algorithms.grpo — NeMo-RL"
[6]: https://docs.nvidia.com/nemo/rl/0.3.0/guides/grpo.html "An in-depth Walkthrough of GRPO in NeMo RL — NeMo-RL"
[7]: https://huggingface.co/docs/trl/main/en/grpo_trainer?utm_source=chatgpt.com "GRPO Trainer"
[8]: https://arxiv.org/html/2312.07551v1?utm_source=chatgpt.com "Language Model Alignment with Elastic Reset"


Below is a “single source of truth” doc for your coding agents. It focuses on the parts that are hard to infer from code alone: the real wedge, what is/isn’t possible, what needs web verification, and the experiment + repo design decisions that matter.

---

# ProRL × NOMA “Reference Reset / KL” Experiment: Success Document

## 0) One-sentence goal

Determine whether (a) you can **remove or materially reduce** the **reference-model logprob inference** cost while preserving ProRL-like stability benefits, and/or (b) whether **optimizer state semantics across resets** matter enough to justify a NOMA-native demonstration.

---

## 1) Core context: what ProRL is actually claiming (and what people misread)

### 1.1 ProRL “reset” is overloaded

* ProRL (paper) says: **when validation stagnates or degrades** they perform a **hard reset** of the **reference model and optimizer**. ([arXiv][1])
* NVIDIA ProRL v2 blog says: they reset the **reference policy every ~200–500 RL steps** (or on KL spikes / stalled validation) and **do not clear optimizer state**. ([NVIDIA Developer][2])

**Implication:** “reset” can mean:

1. Update **reference weights** (π_ref ← π_current)
2. Reset **optimizer state** (clear Adam moments, step counters, scheduler state, etc.)
   ProRL variants do not treat these identically, so your experiment must isolate them.

### 1.2 The *expensive* part is not “resetting”

People sometimes think “resetting” is like a big GPU JIT cache flush. It isn’t.

* Copying weights / swapping pointers is usually bandwidth-bound and not a major FLOP sink.
* The recurrent sink is **reference logprob inference**: running π_ref to compute log π_ref(y|x) for the samples.

Large-scale GRPO stacks explicitly model **reference logprob inference** as a distinct pipeline stage (because it’s significant enough to schedule). ([Red Hat Developer][3])
OpenRLHF explicitly separates Actor/Reference/Reward/Critic across GPUs for utilization reasons. ([GitHub][4])
HuggingFace TRL GRPOTrainer also frames GRPO as “stay close to the reference policy” and includes a KL estimation step (which, in practice, means ref logprobs are needed unless you change the objective). ([Hugging Face][5])

**Therefore:** if you’re optimizing “the cost,” you’re targeting **ref logprob inference**, not the reset memcopy.

---

## 2) The wedge (what is truly non-trivial and worth testing)

There are exactly two “real” wedges:

### Wedge A (algorithmic): eliminate explicit π_ref inference

**Claim to test:** You can preserve most of the stabilizing effect of “moving reference / trust region” while **not** running a separate reference model forward pass.

**Hard constraint (agents must internalize):**
If your objective contains a term that requires (\log \pi_{\text{ref}}(\cdot)) and π_ref is a different model than the one that generated the samples, you can’t get those numbers “for free.” You must:

* compute them (ref forward),
* or have cached them from an earlier time,
* or approximate them by changing the math.

So the only principled path to “no ref forward” is **change the objective** to anchor to something you already have (typically π_old / behavior policy from rollout).

### Wedge B (state semantics): optimizer reset vs keep vs soft-reset matters

**Claim to test:** Even if you keep ref inference, ProRL’s “hard reset of optimizer” vs ProRL v2’s “don’t clear optimizer” implies **optimizer state handling is not settled** and might matter a lot. ([arXiv][1])

This wedge is where NOMA is naturally aligned: NOMA explicitly aims for topology/state changes to be mechanically defined, including optimizer state behavior. ([GitHub][6])

---

## 3) What NOMA contributes (and what it doesn’t)

### 3.1 What NOMA is *actually* strong at (per the repo)

From NOMA’s README:

* Training loops are a **language construct** (`optimize { ... }`)
* Learnable parameters are explicit buffers (alloc / realloc / free)
* Intent: topology changes should be well-defined, including optimizer state behavior
* Jupyter `%%noma` cell magic exists
* I/O includes **Safetensors** ([GitHub][6])

### 3.2 What NOMA does *not* magically solve

* NOMA does not remove information-theoretic requirements: if you need π_ref logprobs, you need to obtain them.
* NOMA’s current status is “alpha”; don’t sell GPU-kernel dominance. The credible claim is: **rapid iteration on state semantics** and “hard-to-do-in-Python” topology/state transforms. ([GitHub][6])

**Translation for agents:** don’t overfit on micro-optimizing tensor ops. Overfit on **clean ablations** and **stateful transforms**.

---

## 4) What needs web search (agents can’t guess this safely)

These are the items that should always be verified with web sources, because details vary by implementation and change quickly:

### 4.1 ProRL/ProRL v2 exact reset triggers and semantics

* Paper: “hard reset when validation stagnates/degrades,” includes optimizer reset. ([arXiv][1])
* Blog: “every 200–500 steps,” optimizer **not** cleared. ([NVIDIA Developer][2])

Agents must not assume these match; your experiment should include both regimes explicitly.

### 4.2 How major stacks compute KL / ref logprobs

Some stacks compute:

* token-level “KL” as `new_logp - ref_logp` on sampled tokens,
* or reward shaping (KL penalty in reward),
* or approximate KL vs π_old.

You need to confirm exact formulation in:

* Async-GRPO (explicit ref logprob inference stage) ([Red Hat Developer][3])
* OpenRLHF (separate reference model, KL penalty) ([GitHub][4])
* TRL GRPOTrainer (reference closeness / KL estimation step) ([Hugging Face][5])
* Any NeMo RL GRPO docs you rely on ([NVIDIA Docs][7])

### 4.3 Whether anyone already tried “no-reference GRPO” publicly

This is the “novelty audit.” Agents should search for:

* “GRPO no reference model”
* “reference free KL penalty”
* “PPO trust region without reference model RLHF”
* “RLOO / RLAIF trust region”
* “KL to behavior policy” in RLHF codebases

If it exists, your contribution is either: (1) better measurement, (2) connecting to ProRL resets, or (3) NOMA state semantics.

### 4.4 X/Twitter notes

The value here is mostly “folklore and engineering notes” (e.g., how resets are implemented operationally). Search terms:

* “ProRL reference reset optimizer”
* “GRPO reference logprob bottleneck”
* “Async-GRPO ref inference”
  But treat X as weak evidence unless linked to code or docs.

---

## 5) The math you should base everything on (sketch-level, but precise)

### 5.1 Common reference-KL penalty (tokenwise on sampled actions)

For samples (y) (tokens/actions), a typical per-token penalty uses:

[
\mathrm{KL}*{\text{sampled}} \approx \mathbb{E}*{t}\Big[\log \pi_\theta(y_t|s_t) - \log \pi_{\text{ref}}(y_t|s_t)\Big]
]

This requires (\log \pi_{\text{ref}}(y_t|s_t)). If π_ref is a separate frozen model, that’s a ref forward pass (unless cached).

### 5.2 “No-reference” trust region (π_old anchor)

If your rollout was generated by π_old, you already have (\log \pi_{\text{old}}(y_t|s_t)) stored from generation (or you can store it).

Then you can impose a trust region penalty:

[
\mathrm{KL}*{\text{old}} \approx \mathbb{E}*{t}\Big[\log \pi_\theta(y_t|s_t) - \log \pi_{\text{old}}(y_t|s_t)\Big]
]

This eliminates π_ref inference, but changes what you’re anchoring to.

**Key equivalence intuition:**
ProRL’s “reset π_ref to current π” makes π_ref a **recent** anchor. Anchoring to π_old (behavior) is also a **recent** anchor. So it is plausible that much of the stabilization is “recent trust region,” not “base SFT anchor.”

**But:** anchoring to π_old does not guarantee you stay near the original base/SFT distribution over long time horizons—unless you add additional mechanisms (periodic anchoring to base, or occasional “outer resets”).

### 5.3 Reference resets: what they’re doing

If you set ( \pi_{\text{ref}} \leftarrow \pi_{\theta_k} ) every K steps, then the KL penalty becomes a local trust region around the most recent anchor, instead of a tight constraint around a stale base model.

Paper rationale: KL term dominates later; resets make it “local again” and keep learning moving. ([arXiv][1])

---

## 6) Experiment scope: what to run (variants/ablations/tasks/loops)

This is the “minimum credible” design that agents should implement.

### 6.1 Variants (must be cleanly separated)

#### Variant A — Explicit reference model KL (baseline)

* Maintain π_ref as a frozen model.
* Compute (\log \pi_{\text{ref}}) on the sampled sequences.
* Apply KL penalty to keep π close to π_ref.
* Implement reference resets per a schedule/trigger.

**Cost note:** you will pay a reference forward unless cached.

#### Variant B — No reference model (the wedge)

* Remove π_ref entirely from the inner loop.
* Store (\log \pi_{\text{old}}) during rollout.
* Apply KL/trust region penalty vs π_old (or PPO-style clipping).

**This is the only principled way to “avoid ref forward” without cheating.**

#### Variant C — Reference model, but reset semantics ablation (NOMA-aligned)

Same as A but with optimizer-state semantics when reference resets happen:

* C1: hard reset optimizer state (paper-style) ([arXiv][1])
* C2: keep optimizer state (ProRL v2 blog-style) ([NVIDIA Developer][2])
* C3: soft reset: scale moments (m \leftarrow \alpha m, v \leftarrow \alpha v), maybe reset step counters / LR warmup

### 6.2 Reset regimes (separate knobs)

Run each variant under:

* **Event-driven** reset: “if validation stagnates / KL spikes” (closest to paper wording) ([arXiv][1])
* **Periodic** reset every K steps (closest to ProRL v2 blog) ([NVIDIA Developer][2])

### 6.3 Ablations (tight, not a huge grid)

* Reset frequency (K \in {200, 500, 1000}) (choose based on your step definition)
* KL coefficient β: two values (low/high)
* Optimizer reset: keep vs hard vs soft (α in {0.1, 0.3})

### 6.4 Tasks (choose verifiable rewards)

You want tasks where:

* reward is deterministic/programmatic (“verifiable”),
* you can measure improvement in <1–2 days,
* the model doesn’t require an RM.

Options:

* arithmetic / short math proofs / string transformations
* unit-testable code completion on tiny snippets
* constrained generation with exact-match scoring

For Tier-1 “toy”: string transforms (reverse digits, bracket matching, checksum) are good because you can get fast signal.

For Tier-2 “credible”: pick a small, standard verifiable dataset and restrict to small completions.

### 6.5 Policy training loops (what matters operationally)

Agents should implement:

* rollout generation
* logprob computation for current policy on tokens
* advantage estimation (GRPO / PPO variants)
* optimization over minibatches for E epochs

Important: in real GRPO/PPO setups, you do **multiple epochs per rollout batch**, so ref logprobs should be computed once per rollout batch and reused—*that’s standard.* The wedge is not “cache across minibatch epochs” (everyone does that). The wedge is eliminating ref inference entirely (Variant B).

---

## 7) Systems measurement: what to log so results are defensible

If your claim touches “reference forward cost,” you must produce systems evidence:

### 7.1 Count forwards

Log per training step:

* `policy_forward_calls`
* `ref_forward_calls`
* `tokens_processed_policy`
* `tokens_processed_ref`

Async-GRPO’s docs explicitly call out ref inference as a stage; mirror that stage-based accounting. ([Red Hat Developer][3])

### 7.2 Wall-clock and throughput

* time per rollout
* time per update epoch
* tokens/sec end-to-end
* GPU utilization if possible

### 7.3 Stability indicators

* KL magnitude vs anchor over time
* entropy collapse
* gradient norm spikes
* “reset events” markers (when resets happen, record reason)

### 7.4 Outcome metrics

* reward / exact-match / pass rate on held-out prompts
* plus “policy drift” metrics (how far from base distribution), if you care about that

---

## 8) What coding agents are likely to get wrong (preempt these)

### 8.1 Confusing “no reference forward” with “cached reference forward”

Caching ref logprobs once per rollout batch is standard; it’s not the wedge.

**Variant B must truly remove π_ref evaluation**, not just reduce repeated calls.

### 8.2 Mis-implementing KL

Token-level KL approximations vary. Agents should implement exactly one or two well-defined versions and keep them fixed across variants. Otherwise results are uninterpretable.

### 8.3 Wrong fairness comparison

Variant B changes the anchor from π_ref to π_old. If π_ref is “SFT base” and π_old is “current policy,” those are different constraints.

To make the comparison fair:

* set π_ref to be **recent** via resets (so A is also a local trust region)
* or add a second outer constraint (e.g., occasional base anchoring) in both A and B

### 8.4 Resetting optimizer state incorrectly

Optimizer “state” isn’t just m,v:

* Adam includes step counts
* schedulers maintain internal state
* mixed precision scalers maintain state

If you “hard reset,” reset all of it.
If you “soft reset,” define exactly what gets scaled vs reset.

### 8.5 Distributed sync

If you reset reference weights in distributed training:

* ensure all ranks update consistently
* ensure checkpoint load/broadcast is measured separately

---

## 9) How to integrate NOMA later (design constraints you should enforce now)

Even if you’re currently in PyTorch, design the repo so that a NOMA integration is mechanical:

### 9.1 Separate “algorithm state” from “engine”

Have a clean separation between:

* rollout data format (tokens, logp_old, rewards, masks)
* update step inputs (params, batch)
* update step outputs (new params, metrics)

### 9.2 Use file-based interchange (lowest friction)

NOMA supports safetensors I/O per its README. ([GitHub][6])
The easiest bridge:

* PyTorch writes a batch and current params to safetensors
* NOMA reads, performs update, writes updated params + metrics

Agents should avoid “tight Python ↔ NOMA live tensor passing” until proven stable.

### 9.3 The NOMA-aligned deliverable

The most “NOMA-shaped” demo is **Variant C**:

* encode optimizer-state transforms (keep/hard/soft) as first-class
* show faster reconvergence / stability benefits when preserving state, consistent with NOMA’s thesis ([GitHub][6])

---

## 10) “Points of support” (what you cite when you write it up)

Use these as your grounding references when writing the eventual report:

* ProRL paper: hard reset reference model + optimizer when validation stalls ([arXiv][1])
* NVIDIA ProRL v2 blog: periodic ref resets every 200–500 steps; optimizer state not cleared ([NVIDIA Developer][2])
* Async-GRPO: explicit “reference log probabilities inference” stage (shows it’s a real systems cost) ([Red Hat Developer][3])
* OpenRLHF: separates Actor/Reference models across GPUs; per-token KL penalty from SFT/reference ([GitHub][4])
* TRL GRPOTrainer docs: GRPO includes KL estimation to remain close to reference policy ([Hugging Face][5])
* NOMA README: optimize blocks, realloc semantics, intent to preserve optimizer state across structural changes, notebook extension ([GitHub][6])

---

## 11) Minimal “definition of done” (so you don’t wander)

Your agents have “success” when you have:

1. A clean implementation of Variants A/B/C under at least one stable task.
2. Systems logs showing:

   * Variant A has nontrivial ref forward cost
   * Variant B eliminates it (ref forwards ≈ 0)
3. A result matrix:

   * performance (reward/accuracy)
   * stability (KL/entropy spikes)
   * cost (tokens/sec, time/step)
4. A clear conclusion:

   * either “no-ref preserves most gains” (Wedge A confirmed),
   * or “optimizer reset semantics matter” (Wedge B confirmed),
   * or both are false (still valuable; it narrows hypotheses).

---

## 12) Practical advice for coordinating multiple agents

Give coding agents a strict contract:

* **Agent 1 (Algo correctness):** implement A/B losses, ensure invariants (masking, logprobs, advantage). Unit tests for logprob and KL formulas.
* **Agent 2 (Systems measurement):** stage timers + forward counters, write JSONL logs, reproducible runs.
* **Agent 3 (Baselines + literature/code audit):** confirm what big stacks do, produce a short “novelty audit” memo with links and exact KL formulas used.
* **Agent 4 (NOMA integration later):** define safetensors schemas and NOMA script I/O; no need to touch main training loop initially.

---

If you come back with results, the next high-leverage thing I can do is: **help you interpret failures** (e.g., why Variant B collapses or diverges) and tighten the math/controls so the writeup is bulletproof.

[1]: https://arxiv.org/html/2505.24864v1?utm_source=chatgpt.com "ProRL: Prolonged Reinforcement Learning Expands ..."
[2]: https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/?utm_source=chatgpt.com "Scaling LLM Reinforcement Learning with Prolonged ..."
[3]: https://developers.redhat.com/articles/2025/04/05/async-grpo-open-fast-and-performant?utm_source=chatgpt.com "Async-GRPO: Open, fast, and performant"
[4]: https://github.com/OpenRLHF/OpenRLHF?utm_source=chatgpt.com "OpenRLHF/OpenRLHF: An Easy-to-use, Scalable and ..."
[5]: https://huggingface.co/docs/trl/main/en/grpo_trainer?utm_source=chatgpt.com "GRPO Trainer"
[6]: https://github.com/pierridotite/Noma "GitHub - pierridotite/NOMA: A systems programming language where automatic differentiation is a compiler pass and model parameters are explicit, growable memory."
[7]: https://docs.nvidia.com/nemo/rl/latest/guides/grpo.html?utm_source=chatgpt.com "An In-depth Walkthrough of GRPO in NeMo RL"


I’ll web-search for ProRL/GRPO-style “reference reset + optimizer reset” discussions, check major RLHF/LLM-RL codebases for how they handle ref/KL and optimizer state, scan X notes, then synthesize a principled wedge + math sketch and a tightly scoped experiment plan.


## 1) The wedge (what we’re actually trying to prove)

If you want a **scoped, non-hand-wavy** NOMA “win” here, it has to be one of these:

1. **You can get most of ProRL’s stability/ablation gains *without* paying the “extra reference forward”** (i.e., eliminate the explicit reference-model logprob pass by changing the formulation in a controlled way).

2. Or, if you *can’t* eliminate it algorithmically, you can at least make **reference resets + optimizer-state handling** (keep / reset / soft-reset) *explicit and cheap to iterate on*, and show that “what happens to optimizer state” matters empirically in this non-stationary RL objective.

NOMA is naturally aligned with (2): it’s explicitly about making topology/state changes mechanically well-defined, including optimizer state behavior. ([GitHub][1])
But your original question is explicitly about (1): **avoiding the extra reference forward**. That requires an algorithmic wedge, not just a systems micro-optimization.

So we’ll treat this as: **Can we remove the reference forward while preserving the “moving anchor” effect that ProRL claims helps?**
If yes → that’s a clean demo. If no → then we pivot to “optimizer-state semantics across resets” as the demo.

---

## 2) What is being reset in ProRL, and what’s actually expensive?

### What ProRL says

In the ProRL paper, the motivation is: the **KL term can dominate** later in training and stall updates, so they do **reference policy reset** and **reinitialize optimizer state** to keep training improving. ([arXiv][2])
They also describe doing a “hard reset of the reference model and optimizer” when validation stagnates or degrades. ([arXiv][2])

So there are two concepts people conflate:

* **Resetting the *reference policy***: set π_ref to a more recent snapshot of π (online policy).
* **Resetting the *optimizer state***: clear Adam moments/etc (m, v, etc), possibly restart LR schedules.

### NVIDIA’s ProRL v2 blog adds a crucial detail

NVIDIA’s ProRL v2 writeup describes resetting the reference to the current policy **every ~200–500 steps**, and explicitly says **the optimizer state is not cleared** in their implementation.
So in the wild, “reset” can mean:

* “update reference” only (cheap-ish), or
* “update reference + optimizer hard reset” (more disruptive), depending on recipe.

### Cost reality check

There are **three different costs** here:

1. **The “reset” operation itself** (updating π_ref; maybe zeroing optimizer):

   * Usually just **memory traffic** (copy weights / swap pointers; maybe memset optimizer buffers).
   * On big distributed setups it can involve **sharded parameter copies** and some coordination, but it’s still *tiny* compared to running hundreds of millions/billions of token-level matmuls.

2. **The *reference forward pass*** (what you called “extra reference forward”):

   * This is the big one. You need log π_ref(a|s) to compute a KL-like penalty (or ratios involving π_ref), which means **running the ref model on the token sequences**.
   * Real cluster implementations treat this as a distinct stage; e.g. async-GRPO explicitly lists “reference model inference (compute ref log probabilities)” as its own phase. ([GitHub][3])
   * OpenRLHF’s system diagram likewise calls out computing **reference policy logprobs** as part of the engine.

3. **The algorithmic disruption cost** (if you reset optimizer):

   * Not “expensive compute,” but it can change convergence/stability.
   * Literature is mixed: resetting optimizer state *can* help in non-stationary RL objectives ([arXiv][4]), but other work finds it often has little effect because moments re-adapt quickly. ([Proceedings of Machine Learning Research][5])

**Bottom line:** if your goal is “avoid the extra cost,” you should be targeting **(2)** (reference forward), not the reset memcopy.

---

## 3) What big codebases do today (and whether anyone already “avoids the ref forward”)

From what’s publicly described in large RLHF/GRPO stacks:

* They **do** compute reference logprobs explicitly (OpenRLHF, async-GRPO). ([GitHub][3])
* Tooling (like VERL) exposes different KL formulations, but it still fundamentally assumes you have access to reference logprobs when you choose a reference-based KL.
* I did not find a widely-used implementation that “magically” gets reference KL without paying reference evaluation; when people say they reduced cost, it’s usually **pipeline overlap, offloading, sharding, or caching**, not elimination.

On X specifically: I didn’t find any deep public notes beyond high-level tweet summaries; nothing like “here’s the trick that removes the ref pass” that I’d trust enough to lean on.

---

## 4) Literature that actually matters for the “reset” story

Two relevant axes:

### A) Why resetting optimizer state might help

ProRL’s own framing is non-stationarity: once you change the effective constraint/anchor, you’ve changed the optimization landscape. ([arXiv][2])
That matches older deep-RL observations: **optimizer moments can become stale/misaligned** across regime changes, and resetting can help. ([arXiv][4])

But you should not assume it’s always helpful:

* Nikishin et al. report resetting optimizer state had essentially no impact in their setting. ([Proceedings of Machine Learning Research][5])

So optimizer reset is an empirical question.

### B) “Soft reset” as a principled alternative

If hard reset is too destructive, there’s work on **soft parameter resets** (don’t throw away everything; partially revert / increase plasticity). ([NeurIPS Proceedings][6])
This is relevant because ProRL v2 explicitly *doesn’t* clear optimizer state when doing frequent ref resets.

---

## 5) Math sketch: what “reference reset” is doing

A simplified view of the common objective:

[
\max_\theta ; \mathbb{E}*{x\sim \mathcal{D},, y\sim \pi*\theta(\cdot|x)}[R(x,y)]
;-; \beta , \mathbb{E}*{x,y}\big[\mathrm{KL}(\pi*\theta(\cdot|x)|\pi_{\text{ref}}(\cdot|x))\big]
]

Tokenwise (for sampled tokens) this often becomes something like:

[
\mathcal{L}*{KL} \propto \mathbb{E}*{t}\big[\log \pi_\theta(y_t|x, y_{<t}) - \log \pi_{\text{ref}}(y_t|x, y_{<t})\big]
]

**Resetting the reference** means: every K steps, set
[
\pi_{\text{ref}} \leftarrow \pi_{\theta_{k}}
]
for some recent checkpoint ( \theta_k ). ProRL’s claim is that this prevents the KL term from becoming an ever-tightening constraint that stalls learning. ([arXiv][2])

**Key point for your “extra reference forward” question:** as long as your KL depends on (\log \pi_{\text{ref}}(\cdot)), you must obtain (\log \pi_{\text{ref}}) on the sequences you train on — which is the reference forward.

So: **you only avoid the reference forward if you change the KL to something that doesn’t require evaluating a separate π_ref**.

---

## 6) Scoped experiment plan (principled, minimal, and actually diagnostic)

### Hypotheses (be explicit)

**H1 (algorithmic wedge):** Most of the benefit of “moving reference resets” comes from enforcing a *recent-policy trust region*, which can be approximated using **behavior-policy logprobs already available from rollout** → eliminating explicit π_ref evaluation.

**H2 (state wedge):** If you do need explicit π_ref, then **how you handle optimizer state at reset** (keep vs hard reset vs soft reset) materially affects stability/learning speed in prolonged RL.

### Experimental setup (two-tier so you don’t overcommit)

#### Tier 1: “One-shot” toy that NOMA can actually run

A contextual bandit or tiny sequence model where:

* policy outputs a categorical distribution
* reward is verifiable (synthetic)
* you can cleanly implement: (i) KL-to-reference, (ii) reference reset, (iii) optimizer reset/soft-reset

This tier exists to validate *mechanics* and let you iterate fast in NOMA.

#### Tier 2: small LLM GRPO run (PyTorch/VERL/TRL) for external credibility

Use a small open model (≤1.5B) on a verifiable-reward dataset (math/code style, small subset) because ProRL is about verifiable rewards. ProRL itself uses VERL + GRPO/DAPO variants. ([arXiv][2])
You’re not reproducing ProRL; you’re testing the wedge.

### Training loops to compare (variants)

#### Variant A: Explicit reference KL (baseline cost)

* Maintain π_ref as a frozen model.
* Compute both logπθ and logπ_ref on training sequences.
* Apply KL penalty.
* Reference reset every K steps (copy θ→ref).

This matches the “extra reference forward exists” world. ([GitHub][3])

#### Variant B: **No reference model** (the wedge)

Replace KL-to-reference with a **trust-region penalty to behavior policy** (π_old), using logprobs you already store from rollout (typical PPO/GRPO-style bookkeeping):

* You still compute logπθ during update.
* You *do not* compute logπ_ref via a second model.
* Penalize divergence from π_old (approx KL or clipping).

This is the only clean way to “avoid the extra reference forward” without pretending physics changed.

What you are testing: *Does this recover most of the stability that ProRL attributes to periodic reference resets?*

#### Variant C: Explicit reference KL + different reset semantics (state wedge)

Keep Variant A’s explicit reference, but change reset behavior:

* C1: **hard reset** optimizer state (as ProRL describes) ([arXiv][2])
* C2: **no optimizer reset** (as ProRL v2 blog describes for frequent ref resets)
* C3: **soft reset** (scale moments down, maybe bump LR briefly), motivated by soft reset literature ([NeurIPS Proceedings][6])

This is where NOMA is philosophically strongest: defining these state transforms explicitly, not as ad hoc trainer hacks. ([GitHub][1])

### Ablations (keep it tight)

* Reset frequency (K): {100, 300, 1000} steps
* KL strength β: {low, medium} (don’t grid search; pick 2)
* Optimizer reset type: {keep, hard, soft}
* (If you’re using DAPO/GRPO extras) lock everything else constant so you’re isolating the wedge.

### Metrics (must include “cost” if your claim is about cost)

You need both **quality** and **systems** metrics:

**Systems**

* wall-clock time per update
* tokens/sec
* % time spent in “ref logprob” stage (Variant A/C only) — async-GRPO explicitly treats this as a stage, so measure similarly ([GitHub][3])

**Stability**

* KL over time
* entropy collapse events
* variance of returns / advantage stats

**Outcome**

* task success metric (pass@k / reward / accuracy) on a fixed eval set

### Success criteria (don’t wiggle later)

* If Variant B gets ≥80–90% of Variant A/C’s final score **and** reduces step time materially (you’ll likely see near-elimination of the ref-logprob stage), that’s a strong wedge: “moving-reference benefit is mostly trust-region; explicit reference forward is optional.”
* If Variant B fails but C2/C3 beat C1, then your wedge becomes: “optimizer reset semantics matter; naive hard reset is not always best; NOMA makes these transforms explicit.”

---

## What you might be “missing” (the blunt version)

* **Resetting the reference weights is not the expensive part.** The expensive part is **evaluating π_ref** on sequences (extra forward), and that cost doesn’t go away unless you change the objective to not require π_ref logprobs. ([GitHub][3])
* The ProRL paper explicitly says they reinitialize optimizer states with reference resets/hard resets, but NVIDIA’s ProRL v2 recipe for frequent resets says they don’t clear optimizer state — so you should treat “reset” as an overloaded term and isolate which one you’re studying. ([arXiv][2])
* If your demo is “NOMA makes copying weights faster,” nobody will care. If your demo is “we removed the reference forward (or proved you can’t) and quantified the tradeoff,” people will care.

If you want, next step (after this outline) is: we pick **one** of the two wedges (algorithmic B vs state C), and I’ll force it into a concrete “minimal runnable” spec (exact loop, exact tensors logged, exact plots).

[1]: https://github.com/pierridotite/Noma "GitHub - pierridotite/NOMA: A systems programming language where automatic differentiation is a compiler pass and model parameters are explicit, growable memory."
[2]: https://arxiv.org/html/2505.24864v1 "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models"
[3]: https://github.com/Red-Hat-AI-Innovation-Team/async-grpo "GitHub - Red-Hat-AI-Innovation-Team/async-grpo"
[4]: https://arxiv.org/abs/2306.17833?utm_source=chatgpt.com "Resetting the Optimizer in Deep RL: An Empirical Study"
[5]: https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf?utm_source=chatgpt.com "The Primacy Bias in Deep Reinforcement Learning"
[6]: https://proceedings.neurips.cc/paper_files/paper/2024/file/978cc34c539fd26f0e8afb7e3905f34a-Paper-Conference.pdf?utm_source=chatgpt.com "Non-Stationary Learning of Neural Networks with Automatic ..."


