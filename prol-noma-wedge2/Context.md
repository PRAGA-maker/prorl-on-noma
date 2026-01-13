Context:
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




Additional Context if Needed:

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


