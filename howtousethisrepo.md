# Multi-Experiment Research Platform

A research platform for rapidly iterating on multiple Noma-based experiments with agentic project management (APM) workflows.

## Overview

This repository enables fast iteration on multiple concurrent research experiments building on top of [Noma](https://github.com/pierridotite/Noma). Each experiment can be managed independently with agentic workflows that support automatic ChatGPT research integration.

## Repository Structure

```
/
├── external/
│   └── noma/              # Shared Noma repository (all experiments use this)
├── experiments/           # All experiments live here
│   ├── prol-noma-wedge2/  # Example experiment
│   └── <exp-name>/        # Your experiments
├── agentcommands/         # APM agent commands and guides
│   ├── .apm/
│   │   ├── guides/        # APM workflow guides
│   │   └── Memory/        # Experiment-scoped memory
│   └── .cursor/
│       └── commands/      # Agent initiation commands
└── README.md              # This file
```

## Quick Start

### 1. Create a New Experiment

Use the experiment management command:
```
.cursor/commands/apm-9-manage-experiment.md
```

Or manually:
```bash
mkdir experiments/my-experiment
cd experiments/my-experiment
# Create experiment structure (see Experiment Template below)
```

### 2. Start an APM Session

1. **Setup Agent**: Initialize with `.cursor/commands/apm-1-initiate-setup.md`
   - Selects/creates experiment
   - Validates Noma integration
   - Conducts context synthesis
   - Creates Implementation Plan

2. **Manager Agent**: Coordinate with `.cursor/commands/apm-2-initiate-manager.md`
   - Receives bootstrap prompt from Setup Agent
   - Manages task assignment and coordination
   - Uses experiment-scoped memory paths

3. **Implementation Agent**: Execute tasks with `.cursor/commands/apm-3-initiate-implementation.md`
   - Receives task assignments
   - Generates ChatGPT prompts for research (automatic)
   - Follows links from ChatGPT responses using browser tools
   - Logs work in experiment-scoped memory

## Key Features

### Multi-Experiment Support
- Each experiment has isolated memory space: `Memory/experiments/<exp-name>/`
- All paths are experiment-relative: `experiments/<exp-name>/`
- Shared Noma repository: All experiments use `external/noma/`

### ChatGPT Integration
- **Automatic Prompt Generation**: Agents automatically generate ChatGPT prompts when research is needed
- **In-Chat Prompts**: Prompts are generated directly in chat as copy-pasteable markdown blocks (never as files)
- **Browser Tool Integration**: Agents automatically follow links from ChatGPT responses
- **Two-Tier Research**: ChatGPT (primary) → Ad-Hoc delegation (fallback)

### ChatGPT Research Workflow

1. **Agent identifies research need** during task execution
2. **Agent generates prompt in-chat** as markdown code block
3. **You copy prompt** from chat
4. **You paste into ChatGPT** and get response (ChatGPT typically provides links)
5. **You return ChatGPT response** to agent
6. **Agent processes response**:
   - Extracts information from text
   - Identifies all URLs/links
   - Uses browser tools to follow links
   - Extracts information from linked pages
7. **Agent integrates findings** into task execution

### Noma Integration

- **Shared Location**: All experiments reference `external/noma/` (relative path: `../../external/noma/`)
- **File-Based Handoff**: Experiments use file-based handoff pattern (see `experiments/<exp-name>/noma/README.md`)
- **Validation**: Setup Agent automatically validates Noma accessibility

## Agent Commands

### Setup Agent (`apm-1-initiate-setup.md`)
- Experiment selection/creation
- Noma integration validation
- Context synthesis
- Implementation Plan creation

### Manager Agent (`apm-2-initiate-manager.md`)
- Experiment-scoped task coordination
- Memory path management: `Memory/experiments/<exp-name>/Phase_XX/`
- Task assignment with experiment-relative paths

### Implementation Agent (`apm-3-initiate-implementation.md`)
- Task execution (single-step or multi-step)
- **ChatGPT prompt generation** (automatic)
- Browser tool integration for following research links
- Experiment-relative file operations

### Research Delegation (`apm-7-delegate-research.md`)
- **ChatGPT-first research protocol** (primary method)
- Ad-Hoc agent delegation (fallback)
- Browser tool usage for link following

### Experiment Management (`apm-9-manage-experiment.md`)
- List experiments
- Create new experiment structure
- Validate experiment setup
- Switch experiment context

## Experiment Template

Standard experiment structure:

```
experiments/<exp-name>/
├── README.md              # Experiment description, quickstart
├── Context.md             # Experiment-specific context (optional but recommended)
├── configs/              # Configuration files
├── src/                   # Source code
├── scripts/               # Utility scripts
├── tests/                 # Tests
├── noma/                  # Noma integration files (optional)
│   └── README.md         # File-based handoff instructions
└── results/               # Experiment outputs (gitignored)
```

## Workflow Example

1. **Create Experiment**:
   ```
   Use apm-9-manage-experiment.md to create "my-experiment"
   ```

2. **Start Setup Agent**:
   ```
   Use apm-1-initiate-setup.md
   - Selects "my-experiment"
   - Validates Noma access
   - Conducts context synthesis
   - Creates Implementation Plan
   ```

3. **Manager Coordinates**:
   ```
   Use apm-2-initiate-manager.md
   - Creates Memory/experiments/my-experiment/Phase_01/
   - Assigns tasks with experiment-relative paths
   ```

4. **Implementation Executes**:
   ```
   Use apm-3-initiate-implementation.md
   - Receives task assignment
   - Needs research → Generates ChatGPT prompt in-chat
   - You copy to ChatGPT, return response
   - Agent follows links, extracts info
   - Completes task, logs to Memory/experiments/my-experiment/Phase_01/
   ```

## Memory System

Memory is scoped to experiments:

```
.apm/Memory/
├── Memory_Root.md                    # Project-level memory
└── experiments/
    ├── prol-noma-wedge2/
    │   ├── Phase_01_Setup/
    │   │   └── Task_1.1_*.md
    │   └── Phase_02_Implementation/
    └── my-experiment/
        └── ...
```

## Best Practices

1. **Experiment Isolation**: Keep experiments independent; use experiment-relative paths
2. **ChatGPT Research**: Let agents generate prompts automatically; you just copy-paste to ChatGPT
3. **Noma Integration**: Use file-based handoff pattern; validate Noma access during setup
4. **Memory Organization**: Each experiment maintains isolated memory space
5. **Fast Iteration**: Create new experiments quickly; agent commands handle context automatically

## Notes

- Everything runs in WSL (Linux environment)
- Noma is shared across all experiments from `external/noma/`
- ChatGPT prompts are always in-chat (never files) for easy copy-paste
- Agents automatically follow links from ChatGPT responses using browser tools

## See Also

- [Noma Repository](https://github.com/pierridotite/Noma)
- Experiment-specific documentation in `experiments/<exp-name>/README.md`
- APM guides in `agentcommands/.apm/guides/`






















# Getting Started: Multi-Experiment Research Platform

A step-by-step guide to using the multi-experiment research platform with APM agent workflows.

## Prerequisites

- This repository cloned and set up
- Noma repository available at `external/noma/` (or you'll set it up)
- Cursor IDE (for agent commands)
- ChatGPT access (for research workflow)

## Step-by-Step Workflow

### Scenario: Starting a New Experiment

Let's say you want to create a new experiment called "my-rl-experiment" to test a reinforcement learning idea with Noma.

---

## Step 1: Create or Select an Experiment

### Option A: Use Experiment Management Command (Recommended)

1. **Open a new chat in Cursor**
2. **Reference the experiment management command**:
   ```
   @agentcommands/.cursor/commands/apm-9-manage-experiment.md
   ```
   Or paste the command file path in chat.

3. **The agent will ask what you want to do**. Say:
   ```
   Create a new experiment called "my-rl-experiment"
   ```

4. **The agent will**:
   - Create `experiments/my-rl-experiment/` directory
   - Set up standard structure (README.md, configs/, src/, etc.)
   - Validate Noma accessibility
   - Report creation status

### Option B: Manual Creation

```bash
# Navigate to repository root
cd /path/to/prorl-on-noma

# Create experiment directory
mkdir -p experiments/my-rl-experiment/{configs,src,scripts,tests,results}

# Copy template
cp experiments/EXPERIMENT_TEMPLATE.md experiments/my-rl-experiment/README.md

# Edit README.md with your experiment details
```

---

## Step 2: Start Setup Agent (Planning Phase)

The Setup Agent creates the Implementation Plan for your experiment.

1. **Open a new chat in Cursor**
2. **Reference the Setup Agent command**:
   ```
   @agentcommands/.cursor/commands/apm-1-initiate-setup.md
   ```
   Or paste: `agentcommands/.cursor/commands/apm-1-initiate-setup.md`

3. **The Setup Agent will greet you** and start the process:
   - **Experiment Selection**: It will detect you're in `experiments/my-rl-experiment/` or ask which experiment
   - **Noma Validation**: It checks that `external/noma/` is accessible
   - **Context Synthesis**: It asks questions about your experiment:
     - What are you trying to test/explore?
     - What's your research question?
     - What do you need to build?
     - What are the technical requirements?
   - **Implementation Plan**: Creates a detailed plan breaking work into tasks

4. **At the end**, Setup Agent gives you a **Bootstrap Prompt** (a markdown code block). **Copy this entire prompt** - you'll need it for Step 3.

---

## Step 3: Start Manager Agent (Coordination Phase)

The Manager Agent coordinates task execution based on the Implementation Plan.

1. **Open a new chat in Cursor** (new session)
2. **Reference the Manager Agent command**:
   ```
   @agentcommands/.cursor/commands/apm-2-initiate-manager.md
   ```

3. **Paste the Bootstrap Prompt** from Step 2 into the chat

4. **The Manager Agent will**:
   - Parse experiment context (name, path, Noma status)
   - Read the Implementation Plan
   - Set up memory structure: `agentcommands/.apm/Memory/experiments/my-rl-experiment/`
   - Create phase directories
   - Start assigning tasks

5. **The Manager Agent will create Task Assignment Prompts** (markdown code blocks). Each prompt is for a specific task.

---

## Step 4: Execute Tasks with Implementation Agent

The Implementation Agent does the actual work (coding, research, etc.).

1. **Open a new chat in Cursor** (new session)
2. **Reference the Implementation Agent command**:
   ```
   @agentcommands/.cursor/commands/apm-3-initiate-implementation.md
   ```

3. **Paste a Task Assignment Prompt** from the Manager Agent (Step 3)

4. **The Implementation Agent will**:
   - Execute the task (coding, file operations, etc.)
   - **If research is needed**: Generate a ChatGPT prompt in-chat (see Step 5)
   - Log work to memory
   - Report completion back to you

5. **Copy the completion report** and return it to the Manager Agent

---

## Step 5: ChatGPT Research Workflow (When Needed)

When the Implementation Agent needs research, it automatically generates a ChatGPT prompt.

### The Process:

1. **Agent generates prompt in-chat** as a markdown code block:
   ```markdown
   # Research Request: Current Noma API Documentation
   
   ## Context
   I'm implementing a file-based handoff between Python and Noma...
   
   ## Research Questions
   1. What is the current Noma API for reading/writing safetensors?
   2. What are the expected tensor formats?
   ...
   ```

2. **You copy the entire prompt** from the chat

3. **You paste it into ChatGPT** (in a ChatGPT window/tab)

4. **ChatGPT responds** (usually with links to documentation)

5. **You copy ChatGPT's response** and paste it back into the Implementation Agent chat

6. **The agent automatically**:
   - Extracts information from the response
   - Identifies links in the response
   - Uses browser tools to follow links
   - Extracts information from linked pages
   - Integrates findings into the task

7. **The agent continues** with the task using the research findings

---

## Complete Example Workflow

Let's trace through a complete example:

### 1. Create Experiment
```
You: "Create experiment 'test-noma-integration'"
[Agent creates structure]
```

### 2. Setup Agent Session
```
You: [References apm-1-initiate-setup.md]
Setup Agent: "Which experiment are you working on?"
You: "test-noma-integration"
Setup Agent: "What are you trying to build?"
You: "A simple RL training loop that uses Noma for the update step"
[... more questions ...]
Setup Agent: [Creates Implementation Plan]
Setup Agent: [Provides Bootstrap Prompt - COPY THIS]
```

### 3. Manager Agent Session
```
You: [References apm-2-initiate-manager.md]
You: [Pastes Bootstrap Prompt]
Manager Agent: [Reads plan, sets up memory]
Manager Agent: [Creates Task Assignment Prompt for Task 1.1 - COPY THIS]
```

### 4. Implementation Agent Session
```
You: [References apm-3-initiate-implementation.md]
You: [Pastes Task Assignment Prompt for Task 1.1]
Implementation Agent: "I need to research the Noma API..."
Implementation Agent: [Generates ChatGPT prompt - COPY THIS]
You: [Copies prompt, pastes to ChatGPT, gets response]
You: [Pastes ChatGPT response back to agent]
Implementation Agent: [Follows links, extracts info, continues task]
Implementation Agent: [Completes task, logs to memory]
Implementation Agent: [Provides completion report - COPY THIS]
```

### 5. Back to Manager Agent
```
You: [Returns to Manager Agent chat]
You: [Pastes completion report]
Manager Agent: [Reviews, creates next Task Assignment Prompt]
[Repeat Steps 4-5 for remaining tasks]
```

---

## Key Points to Remember

### Experiment Context
- **All paths are relative to experiment root**: `experiments/my-rl-experiment/src/model.py` not `/full/path/src/model.py`
- **Memory is experiment-scoped**: `agentcommands/.apm/Memory/experiments/my-rl-experiment/Phase_01/`
- **Noma is shared**: All experiments use `external/noma/` (relative: `../../external/noma/`)

### ChatGPT Research
- **Prompts are in-chat only**: Never files, always copy-pasteable markdown blocks
- **You manually execute**: Copy from agent → Paste to ChatGPT → Return response
- **Agent follows links automatically**: You don't need to open links yourself

### Agent Sessions
- **Each agent type = new chat session**: Setup, Manager, Implementation are separate
- **Copy prompts between sessions**: Bootstrap Prompt, Task Assignments, Completion Reports
- **Memory persists**: All work is logged in `agentcommands/.apm/Memory/`

### Working with Existing Experiments

If you want to continue work on an existing experiment:

1. **Navigate to experiment directory**: `cd experiments/prol-noma-wedge2/`
2. **Start Manager Agent** (or Implementation Agent if you have a task)
3. **Agent automatically detects experiment** from current directory
4. **Continue from where you left off**

---

## Troubleshooting

### "Noma not found"
- Check that `external/noma/` exists
- Verify path from experiment: `../../external/noma/` should work
- Use Experiment Management command to validate setup

### "Experiment not detected"
- Make sure you're in the experiment directory, OR
- Explicitly tell the agent which experiment: "I'm working on experiment X"

### "ChatGPT prompt not generated"
- Research might not be needed for the task
- If you think it is, ask the agent: "This task needs research - generate a ChatGPT prompt"

### "Paths are wrong"
- Remember: all paths are relative to experiment root
- From `experiments/my-exp/`, use `src/model.py` not `experiments/my-exp/src/model.py`

---

## Quick Reference

### Command File Locations
- Setup Agent: `agentcommands/.cursor/commands/apm-1-initiate-setup.md`
- Manager Agent: `agentcommands/.cursor/commands/apm-2-initiate-manager.md`
- Implementation Agent: `agentcommands/.cursor/commands/apm-3-initiate-implementation.md`
- Experiment Management: `agentcommands/.cursor/commands/apm-9-manage-experiment.md`

### Directory Structure
```
experiments/<exp-name>/          # Your experiment
agentcommands/.apm/Memory/experiments/<exp-name>/  # Experiment memory
external/noma/                   # Shared Noma
```

### Workflow Summary
1. Create experiment → 2. Setup Agent (planning) → 3. Manager Agent (coordination) → 4. Implementation Agent (execution) → 5. ChatGPT research (when needed) → Repeat 3-5

---

## Next Steps

- Read `howtousethisrepo.md` for detailed feature documentation
- Check `experiments/EXPERIMENT_TEMPLATE.md` for experiment structure
- See `experiments/prol-noma-wedge2/` for a complete example
- Review APM guides in `agentcommands/.apm/guides/` for advanced usage
