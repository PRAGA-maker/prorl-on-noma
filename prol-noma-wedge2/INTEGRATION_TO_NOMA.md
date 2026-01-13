# How to integrate this into PRAGA-maker/NOMA (branch p-bench)

I can't push or open PRs directly from this environment, but you can integrate in ~2 minutes:

## Option A: drop-in folder

1. In your NOMA repo (branch `p-bench`), create a folder:
   - `bench/prol_noma_wedge/`

2. Copy everything from this repo into that folder.

3. Commit:
   - `git add bench/prol_noma_wedge`
   - `git commit -m "Add ProRL/NOMA wedge toy bench (A/B/C + reset semantics)"`

## Option B: keep as a separate repo

Keep this as a standalone bench repo and point your NOMA documentation/tests to it.

## Notes

- The Python module is under `src/prol_noma_wedge`.
- For quick runs without installation: `PYTHONPATH=src python -m prol_noma_wedge.run ...`
- See `noma/README.md` for the intended file-based handoff boundary.
