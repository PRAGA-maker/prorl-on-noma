# Experiment Structure Guide

This guide describes the standard structure for experiments in this repository.

## Standard Directory Structure

When creating a new experiment, use this structure:

```
experiments/<exp-name>/
├── README.md              # Experiment description, quickstart (REQUIRED)
├── Context.md             # Experiment-specific context for agents (RECOMMENDED)
├── configs/              # Configuration files
│   └── default.yaml      # Example: default configuration
├── src/                   # Source code
│   └── <module>/         # Main Python module
│       ├── __init__.py
│       ├── model.py      # Model definitions
│       ├── trainer.py    # Training logic
│       └── ...
├── scripts/              # Utility scripts
│   ├── plot_results.py   # Analysis/visualization
│   └── run_sweep.py      # Hyperparameter sweeps
├── tests/                # Tests
│   └── test_import.py    # Basic import tests
├── noma/                 # Noma integration files (OPTIONAL)
│   └── README.md         # File-based handoff instructions
└── results/              # Experiment outputs (GITIGNORED)
    └── run1/             # Individual run outputs
```

## Required Files

### README.md
- Experiment overview and goals
- Quickstart instructions
- Running instructions
- Noma integration status
- Results location

Use `EXPERIMENT_TEMPLATE.md` as a starting point.

### Context.md (Recommended)
Experiment-specific context for agents. Should include:
- Research questions and hypotheses
- Technical constraints and requirements
- What needs web verification
- Key design decisions
- Integration requirements

See `experiments/prol-noma-wedge2/Context.md` for an example.

## Optional Directories

### configs/
Configuration files (YAML, JSON, etc.) for different experiment settings.

### src/
Source code for the experiment. Should be a proper Python package if applicable.

### scripts/
Utility scripts for analysis, visualization, or running experiments.

### tests/
Tests for experiment code. Keep tests simple and focused.

### noma/
Noma integration files. Required if experiment uses Noma.

**noma/README.md** should describe:
- File-based handoff interface
- Tensor formats and schemas
- Parameter exchange protocol
- Integration workflow

## Noma Integration

### Shared Noma Location
All experiments reference the shared Noma repository:
- **Relative path**: `../../external/noma/` (from experiment root)
- **Absolute path**: Can be configured if needed

### File-Based Handoff Pattern
Recommended integration pattern:

1. **Python side**: Produce batch tensors, write to disk
2. **Noma side**: Read tensors, run update loop, write parameters
3. **Python side**: Reload parameters, continue

This keeps the interface stable while iterating on update logic in Noma.

### Validation
Setup Agent automatically validates Noma accessibility during experiment setup.

## Path Conventions

### Experiment-Relative Paths
All paths in agent commands and task assignments are relative to the experiment root:
- ✅ `src/model.py` (relative to `experiments/<exp-name>/`)
- ❌ `experiments/<exp-name>/src/model.py` (absolute from repo root)

### Noma Paths
- Use relative path: `../../external/noma/` from experiment root
- Or configure absolute path if needed

## Memory Paths

Agent memory is scoped to experiments:
- Memory location: `.apm/Memory/experiments/<exp-name>/Phase_XX_<slug>/`
- Task logs: `.apm/Memory/experiments/<exp-name>/Phase_XX_<slug>/Task_X.Y_*.md`

## Creating a New Experiment

### Option 1: Use Experiment Management Command
```
Use .cursor/commands/apm-9-manage-experiment.md
- Automatically creates structure
- Validates Noma access
- Sets up initial files
```

### Option 2: Manual Creation
1. Create directory: `experiments/<exp-name>/`
2. Copy structure from this guide
3. Copy `EXPERIMENT_TEMPLATE.md` to `README.md` and fill in
4. Create `Context.md` with experiment-specific context
5. Set up Noma integration if needed

## Best Practices

1. **Isolation**: Keep experiments independent; avoid cross-experiment dependencies
2. **Documentation**: Always include README.md with quickstart
3. **Context**: Provide Context.md for agents to understand experiment specifics
4. **Noma Integration**: Use file-based handoff pattern for clean separation
5. **Results**: Store all outputs in `results/` (gitignored)
6. **Paths**: Always use experiment-relative paths in code and documentation

## Example: Existing Experiment

See `experiments/prol-noma-wedge2/` for a complete example:
- README.md with quickstart
- Context.md with detailed experiment context
- Source code structure
- Noma integration files
- Configuration and scripts
