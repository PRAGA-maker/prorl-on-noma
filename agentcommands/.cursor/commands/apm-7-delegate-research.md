---
priority: 7
command_name: delegate-research
description: Provides the template for delegating a research task to an Ad-Hoc agent
---

# APM 0.5.3 - Research Delegation Guide
This guide defines how Implementation Agents conduct research. **ChatGPT prompt generation is the PRIMARY research method.** Ad-Hoc Research agent delegation is a FALLBACK when ChatGPT research is insufficient.

---

## 1  ChatGPT-First Research Protocol
**CRITICAL**: Always attempt ChatGPT research before Ad-Hoc delegation.

### When to Use ChatGPT Research
Generate ChatGPT prompts automatically when you need:
- Current documentation, APIs, or SDK information
- Literature review (papers, implementations, methodologies)
- Best practices or technical specifications
- Compatibility or integration information
- Experiment planning or hypothesis validation

### ChatGPT Prompt Generation Workflow

1. **Identify Research Need**: Determine that external research is required for task completion
2. **Generate Prompt In-Chat**: Create a copy-pasteable markdown code block directly in your chat response (NEVER create a file)
3. **Present with User Instructions**: Include: "Copy the prompt below and paste it into ChatGPT. Return ChatGPT's response here."
4. **Wait for User Response**: User copies prompt, pastes into ChatGPT, returns response
5. **Process ChatGPT Response**:
   - Parse response text and extract key information
   - **Identify all URLs/links** in ChatGPT response
   - **Use browser tools** (`mcp_cursor-ide-browser_browser_navigate`, `browser_snapshot`) to follow links
   - Extract information from both ChatGPT response and linked pages
6. **Integrate Findings**: Apply research to task execution
7. **Log Research**: Document prompt, response, links followed, and findings in Memory Log

### ChatGPT Prompt Template

```markdown
# Research Request: [Topic]

## Context
[What you are working on and why research is needed]

## Research Questions
1. [Specific question 1]
2. [Specific question 2]
...

## Expected Information
- [Type of information needed]
- [Sources to check (if known)]
- [How information will be used]

## Instructions for ChatGPT
Please provide current information from authoritative sources. Include links to documentation, papers, or codebases when available.
```

### Browser Tool Integration
**MANDATORY**: When ChatGPT provides links:
- Extract all URLs from response
- Use `mcp_cursor-ide-browser_browser_navigate` to open relevant links
- Use `mcp_cursor-ide-browser_browser_snapshot` to capture page content
- Extract information from linked pages
- Integrate information from both ChatGPT response and linked pages

### When to Fall Back to Ad-Hoc Delegation
Proceed to Ad-Hoc delegation (Section 2) only if:
- ChatGPT research (including following links) is insufficient
- Research requires specialized tools or access not available to ChatGPT
- Multiple iterative research cycles are needed with complex follow-up questions
- Research needs exceed ChatGPT's capabilities

---

## 2  Ad-Hoc Delegation Workflow Overview
**Use this only when ChatGPT research is insufficient.**
Ad-Hoc Research agents operate in **separate chat sessions** managed by the delegating Implementation Agent:

### Branch Management
- **Independent Operation**: Ad-Hoc agents work in isolated branched sessions without access to main project context
- **User Coordination**: User opens new chat session, pastes delegation prompt, returns with findings
- **Context Preservation**: Delegation session remains open for potential re-delegation until formal closure

### Handoff Process
1. **Create Prompt**: Use template below with complete research context
2. **User Opens Session**: User initiates new Ad-Hoc Research chat and pastes prompt
3. **Researcher Works**: Ad-Hoc agent investigates sources and provides current information/findings collaborating with User
4. **User Returns**: User brings findings back to Implementation Agent for integration

---

## 2  Delegation Prompt Template
Present delegation prompt **in chat as a single markdown code block with YAML fronntmatter at the top** for User copy-paste to new Ad-Hoc Research session

```markdown
---
research_type: [documentation|api_spec|sdk_version|integration|compatibility|best_practices|other]
information_scope: [targeted|comprehensive|comparative]
knowledge_gap: [outdated|missing|conflicting]
delegation_attempt: [1|2|3|...]
---

# Research Delegation: [Brief Research Topic]

## Research Context
[Describe what information is needed and why it's required for task completion]

## Research Execution Approach
**Primary Goal**: Gather current, authoritative information that Implementation Agents need to proceed with task execution
**Information Delivery Required**: Provide researched documentation, best practices, or technical specifications for Implementation Agent use
**Current Information Focus**: Access official sources and recent documentation rather than providing theoretical guidance
**Knowledge Transfer**: Deliver structured findings that directly answer Implementation Agent questions to enable task continuation

## Research Execution Requirements
**Mandatory Tool Usage**: You must use web search and web fetch tools to access current official documentation and verify information. Do not rely solely on training data or prior knowledge.
**Current Information Standard**: All findings must be sourced from official documentation, GitHub repositories, or authoritative, credible sources accessed during this research session.
**Verification Protocol**: Cross-reference multiple current sources to ensure accuracy and currency of information.

## Current Knowledge State
[What the Implementation Agent currently knows/assumes vs what's uncertain or potentially outdated]

## Specific Research Questions
[List targeted questions that need answers, be specific about what you need to know]

## Expected Sources
[List specific documentation sites, official GitHub repos, API docs, or credible resources for the Ad-Hoc agent to investigate]

## Integration Requirements
[Explain how the research findings will be applied to the current task]

## Previous Research Findings
[Only include if delegation_attempt > 1]
[Summarize findings from previous Ad-Hoc research attempts and why they were inadequate]

## Delegation Execution Note
**Follow your initiation prompt workflow exactly**: Complete Step 1 (scope assessment/confirmation), Step 2 (execution + findings + confirmation request), and Step 3 (final markdown delivery) as separate responses.
```

### Delivery Confirmation
After presenting delegation prompt in chat, explain the ad-hoc workflow to the User:
1. Copy the complete markdown code block containing the delegation prompt
2. Open new Ad-Hoc agent chat session & initialize it with .cursor/commands/apm-4-initiate-adhoc.md
3. Paste delegation prompt to start ad-hoc work
4. Return with findings for integration

---

## 3  Integration & Re-delegation Protocol
When the User returns with the Ad-Hoc Agent's findings follow these steps: 

### Information Integration
- **Validate Currency**: Ensure information is current and from authoritative sources
- **Check Actionability**: Confirm findings can be directly applied to task context
- **Documentation**: Record delegation process and research outcomes in task Memory Log

### Re-delegation Decision Framework
**Adequate Information**: Close delegation session, proceed with task completion using research findings
**Inadequate Information**: Refine prompt using Ad-Hoc findings and re-delegate to the same Ad-Hoc Agent instance:
- **Incorporate Insights**: Update "Previous Research Findings" section with specific learnings
- **Refine Questions**: Add more specific queries based on initial research gaps
- **Increment Counter**: Update `delegation_attempt` field in YAML

### Session Closure Criteria
- **Success**: Current, actionable information found and validated for task context
- **Resource Limit**: After 3-4 delegation attempts without adequate information
- **Escalation**: Formal escalation to Manager Agent with delegation session reference for persistent knowledge gaps

### Memory Logging Requirements
Document in task Memory Log:
- **Research Method**: ChatGPT research OR Ad-Hoc delegation (specify which)
- **Research Rationale**: Why research was needed and what information was required
- **ChatGPT Research** (if used): Prompt generated, response received, links followed, findings extracted
- **Ad-Hoc Delegation** (if used): Session summary, number of attempts, key findings
- **Information Applied**: How research findings were integrated into task completion
- **Session Status**: Closed with adequate information OR escalated with session reference

---

**End of Guide**