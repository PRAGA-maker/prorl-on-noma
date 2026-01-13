# NOMA integration stub

This repo is written in Python first so you can validate the *wedge* quickly.

When you're ready to integrate NOMA, the cleanest approach is file-based handoff:

1. Python produces a batch of tensors for an update step:
   - `input_ids` (B, T)
   - `prompt_T`, `answer_T`
   - `logp_old` (B, answer_T)
   - `rewards` / `advantages`
   - (optional) `ref_logp` (B, answer_T)

2. Write them to disk (e.g., `.pt` or `safetensors`).

3. A NOMA script:
   - reads those tensors
   - runs the update loop (including your chosen reset semantics)
   - writes back updated parameters.

4. Python reloads parameters and continues.

This keeps the interface stable while you iterate on the policy update logic in NOMA.
