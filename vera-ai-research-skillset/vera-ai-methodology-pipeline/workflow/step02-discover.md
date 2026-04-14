# Step 02: Idea Discovery

> **Executor**: Main Agent (invokes sub-skills)
> **Input**: `PIPELINE_STATE.json` (research direction, existing work)
> **Output**: `IDEA_DISCOVERY_REPORT.md` + Gate 1 decision

---

## Execution Instructions

### 2.0 Determine Discovery Assist Mode

Before invoking any sub-skill that can use external reasoning help, inspect the
runtime tool surface:

```text
if "mcp__codex__codex" is in the available tool surface:
    DISCOVERY_ASSIST_MODE = "external"
else:
    DISCOVERY_ASSIST_MODE = "local"
```

Use this mode consistently throughout Step 02:
- `external` -> Codex MCP may assist with brainstorming, novelty cross-checking, and idea review
- `local` -> the main agent performs those tasks directly from search evidence and local reasoning

Step 02 MUST NOT abort just because Codex MCP is unavailable.

### 2.1 Literature Survey

```
Read and execute reference/sub-skills/literature-reviewing.md with context: "{research_direction}"
```

Comprehensive literature search across:
- arXiv (cs.LG, cs.CL, cs.CV, cs.AI, stat.ML)
- Top ML conferences: NeurIPS, ICML, ICLR, AAAI, ACL, EMNLP, CVPR, ECCV
- Google Scholar, Semantic Scholar
- Papers With Code (for benchmark state-of-the-art)
- Local PDFs if available

Output: Literature landscape with key papers, open problems, thematic organization.

### 2.2 Idea Generation

```
Read and execute reference/sub-skills/idea-creating.md with context: "{research_direction}"
```

Systematic brainstorming:
1. Survey methodological landscape (architectures, training strategies, loss functions)
2. External-LLM-assisted brainstorming if available, otherwise direct main-agent brainstorming (8-12 raw ideas)
3. First-pass filtering (feasibility, novelty signal, scope fit)
4. Validation of top 4-6 ideas
5. Pilot experiments for top 2-3 (limited: PILOT_EPOCHS=3, MAX_TOTAL_GPU_HOURS=4)

Output: `IDEA_REPORT.md` with ranked ideas + pilot results.

### 2.3 Novelty Verification

For the top 3 ideas, run:
```
Read and execute reference/sub-skills/novelty-checking.md
```

Multi-source search to verify each idea hasn't been published:
- Extract core claims from each idea
- Search across arXiv, Scholar, conference proceedings, Papers With Code
- Cross-verify with Codex MCP if available; otherwise perform a structured self-cross-check against the closest papers found
- Score novelty 1-10 per idea

### 2.4 Critical Review

```
Read and execute reference/sub-skills/research-reviewing.md
```

Submit top ideas to an external reviewer if available, otherwise perform a
self-critical pre-development review using the same rubric:
- Is the problem well-motivated?
- Is the proposed approach architecturally sound?
- Are there obvious scaling or optimization pitfalls?
- How does it compare to concurrent/recent work?

### 2.5 Compile Discovery Report

**Note on artifact names**: The `idea-creating` sub-skill writes `IDEA_REPORT.md`
(raw brainstorm + pilot results). This step enriches it with novelty scores and
reviewer feedback into `IDEA_DISCOVERY_REPORT.md`. Both files must exist at project
root -- `paper-planning` downstream reads `IDEA_REPORT.md`, and this pipeline reads
`IDEA_DISCOVERY_REPORT.md` for Gate 1. Do NOT rename or overwrite `IDEA_REPORT.md`.

Assemble `IDEA_DISCOVERY_REPORT.md` (project root):

```markdown
# Idea Discovery Report

## Direction
{research_direction}

## Date
{timestamp}

## Literature Landscape
{thematic summary from literature-reviewing}

## Recommended Ideas (Post-Review)

### Idea 1: {Title}
- **Summary**: {one sentence}
- **Hypothesis**: {core conjecture}
- **Minimum viable validation**: {what to implement/benchmark}
- **Novelty score**: X/10
- **Reviewer assessment**: {summary}
- **Risk level**: LOW/MEDIUM/HIGH
- **Estimated effort**: {GPU-days/weeks}
- **Pilot results**: {if available}

### Idea 2: ...
### Idea 3: ...

## Execution Plan
{concrete next steps per idea}
```

### 2.6 GATE 1: Idea Selection

Present top ideas to user:

```
Top ideas from discovery:

1. [Idea 1 title] -- Novelty: X/10, Risk: LOW
   {one-line summary}

2. [Idea 2 title] -- Novelty: X/10, Risk: MEDIUM
   {one-line summary}

3. [Idea 3 title] -- Novelty: X/10, Risk: HIGH
   {one-line summary}

Which idea should we pursue? (default: #1 ranked)
```

**Decision logic**:
```
if AUTO_PROCEED = true:
  wait GATE1_TIMEOUT seconds
  if user responds: use their selection
  else: auto-select #1 ranked
else:
  wait indefinitely for user selection
```

### 2.7 Update State

```json
{
  "stage": 2,
  "status": "completed",
  "discovery_assist_mode": "external",
  "ideas_generated": 10,
  "ideas_validated": 3,
  "selected_idea": {
    "title": "...",
    "summary": "...",
    "hypothesis": "...",
    "novelty_score": 8,
    "risk": "MEDIUM",
    "auto_selected": false
  },
  "timestamp": "..."
}
```

---

## Validation Checkpoints

| ID | Check Item | Pass Criteria | Failure Handling |
|----|------------|---------------|------------------|
| 2a | Literature survey completed | literature-reviewing produced output | Retry; proceed with limited context |
| 2b | Ideas generated | >= 3 validated ideas | Lower threshold; ask user for direction |
| 2c | Novelty checked | Top ideas have novelty scores, with or without Codex MCP | Proceed with caveat |
| 2d | Discovery report written | IDEA_DISCOVERY_REPORT.md exists | Compile from available outputs |
| 2e | Idea selected | selected_idea in state | Wait for user (override auto) |

---

## Next Step
-> Step 03: Implementation (parallel tracks)
