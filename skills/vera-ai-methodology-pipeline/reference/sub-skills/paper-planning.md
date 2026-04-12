# Paper Planning: Claims-Evidence Matrix and Section Outline

Plan the structure of an AI/ML research paper before writing.

## Input

- `IDEA_REPORT.md` or `IDEA_DISCOVERY_REPORT.md` -- selected idea
- `RESULTS_ANALYSIS.md` -- experiment results
- `AUTO_REVIEW.md` -- review feedback (if available)
- Available figures and tables

## Procedure

### Step 1: Extract Claims

From the results and idea, identify all claims the paper will make:

```markdown
| # | Claim | Type | Evidence | Status | Section |
|---|-------|------|----------|--------|---------|
| C1 | Our method outperforms baselines on X | Empirical | Table 2, mean +/- std | SUPPORTED | Results 5.1 |
| C2 | Component A is critical | Empirical | Ablation Table 3 | SUPPORTED | Results 5.2 |
| C3 | Method scales to Y | Empirical | Figure 4 | PARTIAL | Results 5.3 |
| C4 | Approach is more efficient | Empirical | Table 4 (params/FLOPs) | SUPPORTED | Results 5.4 |
| C5 | Method addresses gap Z | Narrative | Lit review | SUPPORTED | Intro 1.3 |
```

### Step 2: Detect Paper Type

| Paper Type | Emphasis | Typical Structure |
|-----------|----------|-------------------|
| Architecture | Model design + ablation | Intro, Related, Method (detailed), Experiments, Analysis |
| Training Strategy | Training method + analysis | Intro, Related, Method, Experiments (many setups) |
| Efficiency | Speed/memory + Pareto | Intro, Related, Method, Efficiency Analysis, Experiments |
| Representation | Probing + downstream | Intro, Related, Method, Probing, Downstream, Analysis |
| Benchmark/Dataset | Dataset construction | Intro, Related, Dataset, Baselines, Analysis |

### Step 3: Section Outline

```markdown
# Paper Plan

## Paper Type: {type}
## Target Venue: {venue}
## Page Budget: {N} pages

## Section Outline

### Abstract (0.25 pages)
- Key claim: {C1}
- Key number: {best metric}

### 1. Introduction (1-1.5 pages)
- Para 1: Context -- {domain significance}
- Para 2: Prior work -- {brief landscape}
- Para 3: Gap -- {what's missing, supports C5}
- Para 4: This paper -- {our contribution}
- Para 5: Summary of results -- {headline numbers}

### 2. Related Work (1-1.5 pages)
- 2.1: {Theme 1}
- 2.2: {Theme 2}
- 2.3: {Theme 3}
- Position: "Unlike X, we Y"

### 3. Method (2-3 pages)
- 3.1: Problem formulation
- 3.2: Architecture / approach overview
- 3.3: Key component (supports C2)
- 3.4: Training procedure
- 3.5: Complexity analysis (supports C4)

### 4. Experiments (2-3 pages)
- 4.1: Setup (datasets, baselines, metrics, seeds)
- 4.2: Main results (Table 2, supports C1)
- 4.3: Ablation study (Table 3, supports C2)
- 4.4: Scaling / efficiency (supports C3, C4)
- 4.5: Analysis (error cases, qualitative examples)

### 5. Conclusion (0.5 pages)
- Summary of contributions
- Limitations
- Future work

### Appendix
- Full hyperparameter tables
- Additional results
- Implementation details
```

### Step 4: Figure and Table Plan

```markdown
## Figures
| # | Type | Content | Section | Source |
|---|------|---------|---------|--------|
| 1 | Architecture | Model diagram | Method 3.2 | Draw new |
| 2 | Training curves | Loss + metric vs epoch | Experiments 4.2 | logs/ |
| 3 | Bar chart | Ablation results | Experiments 4.3 | results/ |
| 4 | Scatter | Efficiency Pareto | Experiments 4.4 | results/ |

## Tables
| # | Content | Section | Source |
|---|---------|---------|--------|
| 1 | Dataset statistics | Setup 4.1 | data/ |
| 2 | Main results | Results 4.2 | RESULTS_ANALYSIS.md |
| 3 | Ablation | Results 4.3 | RESULTS_ANALYSIS.md |
| 4 | Efficiency comparison | Results 4.4 | RESULTS_ANALYSIS.md |
```

## Output

Write `PAPER_PLAN.md` with:
- Claims-evidence matrix
- Section outline with page targets
- Figure and table plan
- Citation plan (key papers to cite per section)

## Key Rules

- Every claim must have evidence (no unsupported claims)
- If evidence is PARTIAL or MISSING, flag it -- either run more experiments or weaken the claim
- Page budget must respect venue limits
- Related Work should position, not just summarize
- Method section should be readable without reading the code
