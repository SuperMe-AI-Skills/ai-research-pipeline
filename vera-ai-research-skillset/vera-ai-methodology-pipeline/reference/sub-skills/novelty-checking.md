# Novelty Checking: AI/ML Research Ideas

Verify that a proposed AI/ML research idea has not already been published.

## Input

A research idea with:
- Title / summary
- Core hypothesis or claim
- Proposed approach (architecture, training strategy, loss function, etc.)

## Procedure

### Step 1: Extract Searchable Claims

From the idea, identify 3-5 core claims that would make the paper novel:
- "We propose [specific mechanism] for [specific task]"
- "We show that [approach] outperforms [standard] on [benchmark]"
- "We demonstrate that [component] is critical for [property]"

### Step 2: Multi-Source Search

For each core claim, search across:

| Source | Query Strategy | What to Look For |
|--------|---------------|------------------|
| arXiv | Title keywords + method name | Direct matches, similar approaches |
| Google Scholar | Exact phrases from claim | Papers using same framing |
| Semantic Scholar | Related paper recommendations | Conceptually similar work |
| Papers With Code | Method name + benchmark | Implementations of same idea |
| Conference proceedings | Search NeurIPS/ICML/ICLR/ACL | Peer-reviewed versions |
| GitHub | Method name + implementation | Open-source implementations |

Use 3+ different query formulations per source.

### Step 3: Cross-Verify with LLM (optional)

First detect whether Codex MCP is available:

```text
if "mcp__codex__codex" is in the available tool surface:
    NOVELTY_REVIEW_MODE = "external"
else:
    NOVELTY_REVIEW_MODE = "local"
```

If `NOVELTY_REVIEW_MODE = "external"`, run:

```
mcp__codex__codex:
  config: {"model_reasoning_effort": "xhigh"}
  prompt: |
    I am checking novelty of this AI/ML research idea:

    Title: {idea title}
    Summary: {idea summary}
    Core claims: {list of claims}

    Here are the closest papers I found:
    {list of closest matches with summaries}

    Questions:
    1. Has this EXACT idea been published? (yes/no/partial)
    2. What is the closest existing work? How does it differ?
    3. What specific aspect, if any, is genuinely novel?
    4. Could a reviewer reasonably reject this as "incremental over [paper X]"?
    5. Novelty score 1-10 (1 = already published, 10 = completely new direction)
```

If `NOVELTY_REVIEW_MODE = "local"`, skip the MCP call and do a structured
self-cross-check from the evidence gathered in Step 2:
- compare each core claim against the closest 2-5 papers
- state whether the overlap is exact / partial / only thematic
- identify the one part that still appears novel, if any
- assign the same 1-10 novelty score and recommendation labels used below

The report format in Step 4 stays the same in both modes.

### Step 4: Score and Report

For each idea, produce:

```markdown
## Novelty Check: {Idea Title}

**Score**: X/10

**Closest existing work**:
1. [Author (Year)] "{Title}" -- [how it overlaps]
2. [Author (Year)] "{Title}" -- [how it overlaps]

**What IS novel**: [specific aspects not found in literature]
**What is NOT novel**: [aspects already covered by existing work]

**Reviewer risk**: [Could a reviewer reject as incremental? Why/why not?]

**Recommendation**: PROCEED / PROCEED WITH CAUTION / PIVOT / ABANDON
```

## Scoring Guide

| Score | Meaning |
|-------|---------|
| 9-10 | Completely new direction, no close matches |
| 7-8 | Novel combination or significant extension of existing work |
| 5-6 | Some novelty but close to existing papers; needs careful positioning |
| 3-4 | Incremental over existing work; high rejection risk |
| 1-2 | Already published or near-identical to existing paper |

## Key Rules

- Search MUST include arXiv (many ML papers are preprint-only)
- Check Papers With Code -- if someone already has a leaderboard entry, it's not novel
- "Novel combination" is valid novelty only if the combination yields new insights
- A concurrent submission (within 6 months) doesn't kill novelty but must be acknowledged
- If novelty score < 5, recommend pivoting to a different angle on the same direction
