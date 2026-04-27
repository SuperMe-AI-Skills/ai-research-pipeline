# Step 06: Manuscript Draft Assembly (LaTeX + PDF)

> **Executor**: Main Agent (invokes `reference/sub-skills/manuscript-writing.md`)
> **Input**: All project artifacts + `AUTO_REVIEW.md` + `RESULTS_ANALYSIS.md`
> **Output**: `paper/main.pdf` + complete `paper/` directory + `RESEARCH_LOG.md`

---

## Execution Instructions

### 6.1 Launch Manuscript Draft Assembly Workflow

```
Read and execute reference/sub-skills/manuscript-writing.md with context: "$ARGUMENTS"
```

This invokes the manuscript-draft assembly workflow which chains 5 sub-skills:

#### Phase 1: Paper Planning (`reference/sub-skills/paper-planning.md`)

Input: AUTO_REVIEW.md, RESULTS_ANALYSIS.md, experiment results, model code
Output: `PAPER_PLAN.md` with:
- Claims-evidence matrix (claim -> type -> evidence -> status -> section)
- Section-by-section outline with page targets
- Figure and table plan
- Citation plan

Paper type detection (from selected idea):
- **Architecture**: Emphasis on model design, ablations. Experiments validate design choices.
- **Training Strategy**: Balance of method + experiments. Ablations isolate training components.
- **Representation Learning**: Emphasis on learned representations. Probing experiments.
- **Efficiency**: Emphasis on FLOPs/latency/memory. Pareto frontier plots.
- **Applied ML**: Emphasis on domain-specific benchmarks. Practical considerations.

Target venues: NeurIPS, ICML, ICLR, ACL, EMNLP, CVPR, AAAI, or journal (JMLR, TMLR).

#### Phase 2: Figure Generation (`reference/sub-skills/figure-creating.md`)

Generate review-ready figures from experiment results:

| Figure Type | Source | Format |
|-------------|--------|--------|
| Architecture diagram | Model code | PDF + PNG |
| Training curves | logs/ | PDF + PNG |
| Main results comparison | results/ | PDF + PNG |
| Ablation bar chart | results/ablations/ | PDF + PNG |
| Data efficiency plot | results/ | PDF + PNG |
| Hyperparameter sensitivity | results/ | PDF + PNG |
| Attention/feature visualization | model outputs | PDF + PNG |

Requirements:
- PDF vector graphics for LaTeX
- 300 DPI PNG raster backup
- Colorblind-safe palettes (use matplotlib colormaps: tab10, Set2)
- LaTeX math in axis labels
- Error bars from multi-seed runs
- Reproducible generation scripts in `paper/figures/gen_*.py`

#### Phase 3: LaTeX Writing (`reference/sub-skills/manuscript-writing.md`)

Venue-specific LaTeX manuscript:

| Section | Content Source | Key Elements |
|---------|---------------|--------------|
| Abstract | Synthesize all sections | 150-250 words, key numbers |
| Introduction | Literature + selected idea | Context, gap, contribution |
| Related Work | Literature survey | Organized by theme, position our work |
| Method | Model code + IDEA_REPORT.md | Architecture, training, loss |
| Experiments | RESULTS_ANALYSIS.md | Setup, main results, ablations |
| Analysis | Ablation + sensitivity | Deep dives, failure cases |
| Conclusion | AUTO_REVIEW.md insights | Summary, limitations, future |
| Appendix | Additional results | Full tables, implementation details |

Writing standards:
- Conference format (NeurIPS: neurips_2026.sty, ICML: icml2026.sty, etc.)
- `\newcommand` for model name, dataset names, metric names
- `\citet{}` / `\citep{}` citations
- Tables with mean +/- std, bold for best results
- All hyperparameters documented in appendix

#### Phase 4: Compilation (`reference/sub-skills/paper-compiling.md`)

Compile LaTeX to PDF:
1. Pre-flight: all `\input` files exist, all figures present
2. Compile: `latexmk -pdf paper/main.tex`
3. Auto-fix errors (up to 3 iterations)
4. Post-check: no `??` references, bibliography renders, page limit respected

#### Phase 5: Writing Improvement (`reference/sub-skills/paper-improving.md`)

2 rounds of writing-focused polish:
- Notation consistency across sections
- Claims match evidence (no overclaiming)
- De-AI artifacts in prose
- Table formatting (bold best, +/- alignment)
- Citation accuracy
- Abstract <-> conclusion alignment
- Page limit compliance (trim if over)

### 6.2 Generate Research Log

After paper pipeline completes, write `RESEARCH_LOG.md`:

```markdown
# Methodology Research Pipeline -- Execution Log

## Pipeline Metadata
- **Research Direction**: {from PIPELINE_STATE}
- **Selected Idea**: {title} (novelty: {score}/10)
- **Idea Selection**: {auto/manual}, Gate 1

## Stage Progression
| Stage | Status | Duration | Notes |
|-------|--------|----------|-------|
| 1. Intake | Completed | -- | {environment summary} |
| 2. Idea Discovery | Completed | ~{X} min | {N} ideas -> {N} validated -> selected #{N} |
| 3. Implementation | Completed | ~{X} min | Tracks: {list} |
| 4. Experiments | Completed | ~{X} hours | {N} runs, {N} datasets, {N} seeds |
| 5. External Review | Completed | ~{X} hours | {N} rounds, final: {score}/10 |
| 6. Manuscript Draft Assembly | Completed | ~{X} min | {pages} pages, {venue} format |

## Key Results
- {Finding 1}
- {Finding 2}
- {Finding 3}

## Review Score Progression
Round 1: {score}/10 -> Round 2: {score}/10 -> ... -> Final: {score}/10

## Deliverables
- `paper/main.pdf` -- Manuscript
- `paper/main.tex` -- LaTeX source
- `src/` -- Model implementation
- `baselines/` -- Baseline implementations
- `results/` -- Experiment outputs
- `IDEA_DISCOVERY_REPORT.md` -- Idea exploration
- `AUTO_REVIEW.md` -- Review loop log
- `PAPER_PLAN.md` -- Paper outline

## Remaining Work for Author
- [ ] Verify all experimental results for correctness
- [ ] Run additional seeds if variance is high
- [ ] Check paper for remaining [TODO] markers
- [ ] Confirm claims match evidence exactly
- [ ] Add acknowledgments, funding, author affiliations
- [ ] Prepare supplementary materials (code release, model weights)
- [ ] Package reproducibility bundle (code + configs + seeds)
- [ ] Check page limit compliance for target venue
- [ ] Review and approve before submission
```

### 6.3 Update Final State

```json
{
  "stage": 6,
  "status": "completed",
  "paper": {
    "venue": "neurips",
    "pages": 9,
    "compile_status": "success",
    "improvement_rounds": 2,
    "final_improvement_score": 7.5
  },
  "total_pipeline_hours": 52.0,
  "timestamp": "..."
}
```

---

## Validation Checkpoints

| ID | Check Item | Pass Criteria | Failure Handling |
|----|------------|---------------|------------------|
| 6a | PAPER_PLAN.md exists | Claims-evidence matrix complete | Regenerate |
| 6b | Figures generated | PDF + PNG for each planned figure | Regenerate missing |
| 6c | LaTeX sections complete | All \input files exist | Regenerate missing section |
| 6d | PDF compiles | paper/main.pdf exists, no errors | Auto-fix, retry (3x) |
| 6e | Page limit respected | Within venue limit (e.g., 9 pages for NeurIPS) | Trim content |
| 6f | Bibliography complete | All citations resolve | Fix .bib |
| 6g | RESEARCH_LOG.md written | Complete execution trace | Generate from state |

---

## Workflow Complete

The methodology research workflow has produced:
- A candidate AI/ML method with full implementation
- Experimental evaluation with ablations
- Statistical significance testing across multiple seeds
- A review-ready LaTeX manuscript draft formatted for a target venue
- Complete reproducibility bundle (code + configs + data)

**Critical reminder**: All experimental claims, novelty positioning, and venue suitability MUST be verified by the human author. The workflow produces a DRAFT; submission and authorship judgment remain human responsibilities.
