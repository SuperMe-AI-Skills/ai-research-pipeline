# AI/ML Research Idea Creator

Generate publishable AI/ML research ideas for: $ARGUMENTS

## Overview

Given a broad research direction, systematically generate, validate, and rank concrete AI/ML research ideas. This skill composes with `reference/sub-skills/literature-reviewing.md`, `reference/sub-skills/novelty-checking.md`, and `reference/sub-skills/research-reviewing.md` to form a complete idea discovery pipeline.

## Constants

- **PILOT_MAX_MINUTES = 30** -- Skip any pilot experiment estimated to take > 30 minutes. Flag as "needs full training".
- **PILOT_TIMEOUT_MINUTES = 45** -- Hard timeout: kill pilots exceeding 45 minutes.
- **MAX_PILOT_IDEAS = 3** -- Pilot at most 3 ideas in parallel.
- **MAX_TOTAL_GPU_HOURS = 4** -- Total GPU budget for all pilot experiments combined.
- **PILOT_EPOCHS = 3** -- Quick pilot uses fewer epochs than full training.
- **REVIEWER_MODEL = `gpt-5.4`** -- Model used when Codex MCP is available for brainstorming and review.

## Workflow

### Phase 1: Landscape Survey (5-10 min)

Map the research area to understand what exists and where the gaps are.

1. **Scan local paper library first**: Check `papers/` and `literature/` for existing PDFs.

2. **Search recent literature** using WebSearch:
   - Top ML conferences in the last 2 years: NeurIPS, ICML, ICLR, ACL, CVPR
   - arXiv cs.LG, cs.CL, cs.CV, stat.ML preprints (last 12 months)
   - Papers With Code leaderboards for relevant benchmarks
   - Use 5+ different query formulations

3. **Build a landscape map**:
   - Group by approach (architecture family, training paradigm, loss type)
   - Identify what has been tried and what hasn't
   - Note recurring limitations in discussion/future work sections
   - Flag open problems explicitly stated by multiple papers
   - Note which scales have been studied (small, medium, large)

4. **Identify structural gaps**:
   - Architectures that work for task A but haven't been tried on task B
   - Training strategies lacking theoretical understanding
   - Methods that scale poorly and need efficiency improvements
   - Promising ideas tested only on one benchmark/domain
   - Missing connections between subfields (e.g., NLP technique for vision)
   - Approaches that work empirically but lack analysis of WHY

### Phase 2: Idea Generation (external LLM if available, otherwise local brainstorm)

First detect whether Codex MCP is available:

```text
if "mcp__codex__codex" is in the available tool surface:
    BRAINSTORM_MODE = "external"
else:
    BRAINSTORM_MODE = "local"
```

If `BRAINSTORM_MODE = "external"`, use Codex MCP for divergent thinking:

```
mcp__codex__codex:
  model: REVIEWER_MODEL
  config: {"model_reasoning_effort": "xhigh"}
  prompt: |
    You are a senior ML researcher brainstorming research ideas.

    Research direction: [user's direction]

    Here is the current landscape:
    [paste landscape map from Phase 1]

    Key gaps identified:
    [paste gaps from Phase 1]

    Generate 8-12 concrete AI/ML research ideas. For each idea:
    1. One-sentence summary
    2. Core hypothesis (what you expect to show and why)
    3. Minimum viable validation:
       - For architecture ideas: small-scale experiment (CIFAR/GLUE subset, 3 epochs)
       - For training ideas: pilot on standard benchmark showing improvement
       - For theoretical ideas: proof sketch of key lemma
    4. Expected contribution type: new architecture / training strategy / loss function / analysis / efficiency method / benchmark
    5. Risk level: LOW / MEDIUM / HIGH
    6. Estimated compute: GPU-hours for full evaluation

    Prioritize ideas that are:
    - Novel but grounded in existing understanding
    - Testable with moderate compute (single GPU, days not weeks)
    - Likely to produce clean, interpretable results
    - Addressing genuine gaps, not incremental tuning
    - Generalizable across tasks/domains

    ML-specific criteria:
    - Does the method have a clear advantage over existing approaches?
    - Is there an ablation study that would cleanly isolate the contribution?
    - Can the approach be fairly compared using standard benchmarks?
    - Is the compute budget reasonable for the expected improvement?
```

If `BRAINSTORM_MODE = "local"`, generate the same 8-12 idea records yourself
from the Phase 1 landscape map. Keep the same schema (summary, hypothesis,
minimum validation, contribution type, risk, compute estimate) and apply the
same ranking criteria. The skill must still produce `IDEA_REPORT.md` even
without Codex MCP.

### Phase 3: First-Pass Filtering

For each generated idea, quickly evaluate:

1. **Feasibility check**: Can we validate with available resources?
   - GPU compute requirements (estimate training time)
   - Dataset availability (public benchmarks preferred)
   - Implementation complexity (can build on existing code?)

2. **Novelty quick-check**: 2-3 targeted searches per idea

3. **Impact estimation**: Would an ML reviewer care?
   - "So what?" test: if the experiment succeeds, does it change practice?
   - Does it connect to active areas (efficiency, foundation models, alignment, robustness)?
   - Is the improvement likely to be significant and consistent?

Eliminate ideas that fail. Typically 8-12 ideas reduce to 4-6.

### Phase 4: Deep Validation (for top ideas)

1. **Novelty check**: Run `reference/sub-skills/novelty-checking.md`
2. **Critical review**: Use Codex MCP if available; otherwise run a self-critical devil's-advocate pass with the same rubric
3. **Combine rankings**: Merge assessments, select top 2-3 for pilot

### Phase 5: Pilot Experiments (for top 2-3 ideas)

1. **Design pilots**: Minimal experiment per idea:
   - **Architecture ideas**: Train on small dataset (CIFAR-10, GLUE subset), PILOT_EPOCHS, compare with standard baseline
   - **Training ideas**: Apply to standard model, compare with default training
   - **Loss ideas**: Swap loss function, measure impact on standard task
   - Clear success metric upfront

2. **Deploy pilots**: Write Python scripts and run:
   ```python
   # Pilot experiment template
   set_seed(42)
   model = ProposedModel(config="debug")
   train(model, debug_dataloader, epochs=PILOT_EPOCHS)
   result = evaluate(model, val_dataloader)
   print(f"Pilot result: {result}")
   ```

3. **Collect and compare results**: Positive/negative/weak signal per idea

### Phase 6: Output -- Ranked Idea Report

Write `IDEA_REPORT.md`:

```markdown
# AI/ML Research Idea Report

**Direction**: [user's research direction]
**Generated**: [date]
**Ideas evaluated**: X generated -> Y survived filtering -> Z piloted -> W recommended

## Landscape Summary
[3-5 paragraphs on the field's current state]

## Recommended Ideas (ranked)

### Idea 1: [title]
- **Hypothesis**: [one sentence]
- **Minimum validation**: [concrete experiment description]
- **Expected outcome**: [what success/failure looks like]
- **Novelty**: X/10 -- closest work: [paper]
- **Feasibility**: [compute cost, implementation complexity]
- **Risk**: LOW/MEDIUM/HIGH
- **Contribution type**: architecture / training / loss / analysis / efficiency
- **Pilot result**: [POSITIVE: +2.3% accuracy / NEGATIVE: no improvement / SKIPPED]
- **Key ablation to run**: [what component to ablate]
- **Reviewer's likely objection**: [strongest counterargument]
- **Target venue**: [NeurIPS / ICML / ICLR / ACL / other]
- **Why we should do this**: [1-2 sentences]

### Idea 2: [title]
...

## Eliminated Ideas (for reference)
| Idea | Reason eliminated |
|------|-------------------|
| ... | Already done by [paper] |
| ... | Requires > 100 GPU-hours to validate |

## Pilot Experiment Results
| Idea | Metric | Baseline | Proposed | Delta | Signal |
|------|--------|----------|----------|-------|--------|
| ... | Accuracy | 85.2% | 87.5% | +2.3% | POSITIVE |

## Suggested Execution Order
1. Start with Idea 1 (positive pilot, clean ablation story)
2. Idea 2 as backup
3. Idea 3 deprioritized

## Next Steps
- [ ] Full training on standard benchmarks (3+ seeds)
- [ ] Complete ablation study
- [ ] Compare with all standard baselines
```

## Key Rules

- The user provides a DIRECTION, not an idea. Your job is to generate ideas.
- Quantity first, quality second: brainstorm broadly, then filter ruthlessly.
- A clean negative result (showing a popular belief is wrong) is publishable.
- Don't fall in love with any idea before validating it.
- Always estimate GPU cost for full experiments, not just pilot.
- "Apply X to Y" is low-value unless it reveals surprising behavior.
- Include eliminated ideas in report -- they save future time.
- **If direction is too broad (e.g., "deep learning", "NLP"), STOP and ask to narrow.**
