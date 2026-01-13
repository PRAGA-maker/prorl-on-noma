# Experiment: <Experiment Name>

## Overview

[One-sentence description of what this experiment tests or explores]

## Goals

- [Primary research question or hypothesis]
- [Secondary goals or questions]
- [Expected outcomes or deliverables]

## Quickstart

### Prerequisites
- Noma accessible at `../../external/noma/` (or absolute path)
- [Other dependencies or requirements]

### Setup
```bash
# Install dependencies
pip install -r requirements.txt  # if applicable
pip install -e .  # if applicable

# Verify Noma integration
ls ../../external/noma/  # Should show Noma repository
```

### Running
```bash
# [Example command to run the experiment]
python -m <module>.run --outdir results/run1 --variant A --steps 500
```

## Experiment Structure

```
experiments/<exp-name>/
├── README.md              # This file
├── Context.md             # Experiment-specific context for agents
├── configs/              # Configuration files
│   └── default.yaml      # Example config
├── src/                   # Source code
│   └── <module>/         # Main module
├── scripts/              # Utility scripts
│   └── plot_results.py   # Example analysis script
├── tests/                # Tests
├── noma/                 # Noma integration (if applicable)
│   └── README.md         # File-based handoff instructions
└── results/              # Experiment outputs (gitignored)
    └── run1/             # Example run output
```

## Noma Integration

**Status**: [Not Started / In Progress / Complete]

If this experiment uses Noma:

- **Integration Type**: [File-based handoff / Direct integration / Other]
- **Noma Location**: `../../external/noma/` (relative) or `[absolute path]`
- **Integration Pattern**: See `noma/README.md` for file-based handoff details

### File-Based Handoff (Recommended)

1. Python produces batch tensors for update step
2. Write to disk (`.pt` or `safetensors`)
3. NOMA script reads, runs update loop, writes updated parameters
4. Python reloads parameters and continues

See `noma/README.md` for detailed interface specification.

## Variants / Ablations

[If applicable, describe different variants or ablations being tested]

- **Variant A**: [Description]
- **Variant B**: [Description]
- **Variant C**: [Description]

## Results

[Link to results or describe where results are stored]

Results are stored in `results/` directory (gitignored). Each run creates a subdirectory with:
- Training logs
- Model checkpoints (if applicable)
- Metrics and plots

## Context for Agents

See `Context.md` for experiment-specific context that agents need:
- Research questions and hypotheses
- Technical constraints
- Integration requirements
- What needs web verification
- Key design decisions

## References

- [Related papers or documentation]
- [Noma documentation](https://github.com/pierridotite/Noma)
- [Other relevant resources]

## Notes

[Any additional notes, known issues, or future work]
