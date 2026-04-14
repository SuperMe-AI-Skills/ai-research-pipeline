# Research Reviewing: Critical Assessment of AI/ML Ideas

Provide critical review of a proposed AI/ML research idea before committing to full development.

## Input

A research idea with:
- Summary and hypothesis
- Proposed approach
- Preliminary results (if available)
- Literature context

## Procedure

### Step 1: Review the Idea (external if available, otherwise self-review)

First detect whether Codex MCP is available:

```text
if "mcp__codex__codex" is in the available tool surface:
    IDEA_REVIEW_MODE = "external"
else:
    IDEA_REVIEW_MODE = "self"
```

If `IDEA_REVIEW_MODE = "external"`, submit to Codex MCP:

```
mcp__codex__codex:
  model: gpt-5.4
  config: {"model_reasoning_effort": "xhigh"}
  prompt: |
    You are a senior ML researcher at a top lab (Google DeepMind, Meta FAIR, or Anthropic).
    You are reviewing a proposed research idea BEFORE the authors commit to full development.

    === PROPOSED IDEA ===
    {idea summary, hypothesis, approach}

    === PRELIMINARY RESULTS (if any) ===
    {pilot results}

    === LITERATURE CONTEXT ===
    {landscape summary, closest work}

    Provide a critical pre-development assessment:

    1. **Problem motivation**: Is this problem worth solving? Who benefits?
    2. **Technical soundness**: Is the proposed approach likely to work? Any fundamental issues?
    3. **Experimental design**: Are the planned experiments sufficient to validate the claims?
    4. **Comparison fairness**: Are the proposed baselines appropriate and sufficient?
    5. **Scalability**: Will this work at scale, or only on toy problems?
    6. **Novelty**: Is this genuinely new, or a known technique in different clothing?
    7. **Presentation risk**: Could this be misunderstood or misrepresented?

    For each point, give:
    - Assessment (STRONG / ADEQUATE / WEAK / CRITICAL)
    - Specific concern (if not STRONG)
    - Suggested fix (if applicable)

    Overall recommendation:
    - PROCEED: Strong idea, go ahead
    - PROCEED WITH CHANGES: Good idea, but fix [specific issues] first
    - RECONSIDER: Significant concerns, pivot may be needed
    - ABANDON: Fundamental issues, not worth pursuing

    Be constructively harsh. Better to kill a bad idea early than waste weeks on it.
```

If `IDEA_REVIEW_MODE = "self"`, use the same rubric locally:
- read the idea, pilot results, and literature context fresh
- assess each of the seven dimensions as STRONG / ADEQUATE / WEAK / CRITICAL
- write the same table and recommendation labels as the external path
- be adversarial and explicit about kill criteria

The skill must always produce the structured review from Step 2, regardless of
whether Codex MCP is installed.

### Step 2: Parse and Structure Review

Extract structured feedback:

```markdown
## Pre-Development Review

**Overall**: PROCEED / PROCEED WITH CHANGES / RECONSIDER / ABANDON

| Dimension | Assessment | Concern | Fix |
|-----------|-----------|---------|-----|
| Problem motivation | STRONG/ADEQUATE/WEAK/CRITICAL | ... | ... |
| Technical soundness | ... | ... | ... |
| Experimental design | ... | ... | ... |
| Comparison fairness | ... | ... | ... |
| Scalability | ... | ... | ... |
| Novelty | ... | ... | ... |
| Presentation risk | ... | ... | ... |

**Critical issues** (must fix before proceeding):
1. ...

**Suggested improvements** (should fix):
1. ...

**Minor notes** (nice to have):
1. ...
```

### Step 3: Integrate with Pipeline

Feed review results back to the idea discovery pipeline:
- If PROCEED: continue to implementation
- If PROCEED WITH CHANGES: apply fixes, optionally re-review
- If RECONSIDER: present alternatives to user, ask for direction
- If ABANDON: remove from candidate list, explain why

## Key Rules

- External review is advisory, not binding -- user makes final decision
- Always present the full review to the user, not a summary
- If reviewer identifies a fundamental flaw, flag it prominently
- One review is sufficient for pre-development; save additional rounds for post-experiment
