# Multi-Experiment Research Platform

A research platform for rapidly iterating on multiple Noma-based experiments with agentic project management (APM) workflows.

## Quick Start

**New to this?** → Read **[GETTING_STARTED.md](GETTING_STARTED.md)** for a step-by-step walkthrough.

**Want details?** → Read **[howtousethisrepo.md](howtousethisrepo.md)** for comprehensive documentation.

## What This Is

This repository lets you:
- Run multiple research experiments concurrently
- Use agentic workflows (APM) to plan and execute experiments
- Automatically integrate ChatGPT for research (you copy-paste prompts)
- Build experiments on top of [Noma](https://github.com/pierridotite/Noma)

## Typical Workflow

1. **Create experiment** → Use `apm-9-manage-experiment.md`
2. **Plan experiment** → Use `apm-1-initiate-setup.md` (Setup Agent)
3. **Coordinate tasks** → Use `apm-2-initiate-manager.md` (Manager Agent)
4. **Execute work** → Use `apm-3-initiate-implementation.md` (Implementation Agent)
5. **Research** → Agent generates ChatGPT prompts, you copy-paste to ChatGPT

## Key Features

- **Multi-experiment**: Each experiment has isolated memory and paths
- **ChatGPT integration**: Agents auto-generate research prompts (you execute them)
- **Shared Noma**: All experiments use `external/noma/`
- **Experiment-scoped**: All paths relative to experiment root

## Structure

```
experiments/              # Your experiments
agentcommands/           # APM agent commands
external/noma/           # Shared Noma repository
```

See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed instructions.