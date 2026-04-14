# Paper Planning: Application Paper Structure

Plan the structure of an applied AI/ML research paper.

## Input

- `output/manuscript.md` or `NARRATIVE_REPORT.md` -- analysis results
- `RESULTS_ANALYSIS.md` -- experiment results
- `output/literature_review.md` -- literature context
- `output/analysis_strategy.md` -- model tracks used

## Procedure

### Step 1: Extract Claims

From results, identify all claims:

```markdown
| # | Claim | Type | Evidence | Status | Section |
|---|-------|------|----------|--------|---------|
| C1 | Model X outperforms baselines on task | Empirical | Table 2 | SUPPORTED | Results |
| C2 | Pre-training helps for this domain | Empirical | Ablation Table | SUPPORTED | Results |
| C3 | Feature Y is most predictive | Empirical | SHAP/importance | SUPPORTED | Analysis |
| C4 | Classical ML is competitive for small data | Empirical | Table 2 | SUPPORTED | Discussion |
| C5 | This task benefits from multi-model analysis | Narrative | Cross-model table | SUPPORTED | Discussion |
```

### Step 2: Section Outline

```markdown
# Paper Plan

## Target Venue: {venue}
## Page Budget: {N} pages

### Abstract (0.25 pages)
- Task + dataset
- Best model + metric
- Key insight

### 1. Introduction (1-1.5 pages)
- Domain context and significance
- What is known from prior work
- Gap: why existing approaches are insufficient
- This study: multi-model analysis approach
- Contribution summary

### 2. Related Work (1-1.5 pages)
- Theme 1: [Domain methods]
- Theme 2: [ML/DL approaches for this task]
- Theme 3: [Comparison studies]

### 3. Data and Experimental Setup (1-1.5 pages)
- 3.1: Dataset description
- 3.2: Input representation
- 3.3: Protocol (splits, metrics, seeds, hardware)
- Table 1: Dataset statistics

### 4. Methods (1.5-2.5 pages)
- 4.1: Classical baselines
- 4.2: Neural models
- 4.3: Pre-trained models
- 4.4: Training details
- 4.5: Ensemble (if applicable)

### 5. Results (2-3 pages)
- 5.1: Main comparison (Table 2)
- 5.2: Ablation / analysis
- 5.3: Error analysis
- 5.4: Efficiency comparison

### 6. Discussion (1-1.5 pages)
- Key findings
- Comparison with literature
- Strengths and limitations
- Future directions

### Appendix
- Hyperparameter tables
- Additional per-class results
```

### Step 3: Figure and Table Plan

```markdown
## Tables
| # | Content | Section |
|---|---------|---------|
| 1 | Dataset statistics | Data 3.1 |
| 2 | Main results (all models) | Results 5.1 |
| 3 | Ablation / component analysis | Results 5.2 |
| 4 | Efficiency comparison | Results 5.4 |

## Figures
| # | Type | Section |
|---|------|---------|
| 1 | Data distribution / examples | Data 3.1 |
| 2 | Training curves | Results 5.1 |
| 3 | Confusion matrix / error analysis | Results 5.3 |
| 4 | Feature importance / SHAP | Results 5.3 |
```

## Output

Write `PAPER_PLAN.md` with claims-evidence matrix, section outline, and figure/table plan.

## Key Rules

- Every claim must have evidence
- Application papers need strong related work sections
- Error analysis is highly valued at ML venues
- Include practical insights (when to use which model)
- Page budget must respect venue limits
