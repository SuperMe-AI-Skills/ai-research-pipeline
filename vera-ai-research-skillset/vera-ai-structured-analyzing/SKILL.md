---
name: vera-ai-structured-analyzing
description: >-
  Server-side extension that completes the full analysis pipeline for
  structured/tabular data after vera-ai-structured-testing has run. Adds SVM,
  Random Forest, XGBoost, LightGBM, CatBoost classifiers/regressors with
  missing value imputation, encoding, and scaling, subgroup analysis by
  metadata variables, deep learning models (MLP, TabNet with optional
  hyperparameter search), stacking ensemble with meta-learner, cross-method
  comparison with unified feature importance on a 0-100 scale, and
  review-ready methods.md and results.md drafts. Supports BOTH classification
  (F1/AUC) AND regression (RMSE/R2/MAE). Applies output variation and
  code style diversity for natural, non-repetitive output. Open-source
  skill. Triggered after vera-ai-structured-testing completes, or direct
  API call with a structured data task.
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob
---

# Structured/Tabular Data --- Full Analysis & Review-Ready Manuscript Drafting

Open-source skill. Read `reference/specs/output-variation-protocol.md`
before every generation --- apply all variation layers for natural, diverse output.

## Workflow

Continues from where vera-ai-structured-testing stopped (PART 0-2 done).

| Step | File | Executor | Output |
|---|---|---|---|
| Additional ML models | `workflow/04-run-additional-models.md` | Main Agent | PART 3 code + prose |
| Subgroup | `workflow/05-analyze-subgroups.md` | Main Agent | PART 4 code + prose |
| Deep learning | `workflow/06-fit-advanced-models.md` | Main Agent | PART 5 code + prose |
| Comparison | `workflow/07-compare-models.md` | Main Agent | PART 6 code + prose |
| Manuscript | `workflow/08-generate-manuscript.md` | Main Agent | methods.md + results.md |

## Additional Inputs

Collect if not already provided:
- Target discipline (for reporting conventions)
- Target journal or style (JMLR, NeurIPS, ICML, domain journal, etc.)
- Research question / hypothesis
- Task type: classification or regression
- Subgroup variable for stratification

## Output Structure

```
output/
├── methods.md
├── results.md
├── tables/             <- Markdown + CSV per table
├── figures/            <- PNGs, 300 DPI
├── references.bib
└── code.py             <- Style-varied
```

## Key References (read before generation)

| File | Purpose |
|---|---|
| `reference/specs/output-variation-protocol.md` | Output quality variation layers |
| `reference/specs/code-style-variation.md` | Seven-dimension code style diversity |
| `reference/patterns/sentence-bank.md` | 4-6 phrasings per result type |
| `reference/rules/reporting-standards.md` | Hard rules for ML/DL reporting |

## Reporting Standards

Same as vera-ai-structured-testing, plus:
- Classification: report F1 (weighted) and AUC (macro) with bootstrapped 95% CIs
- Regression: report RMSE, R2, and MAE with bootstrapped 95% CIs
- Deep learning: report training epochs, best epoch, learning rate, batch size
- TabNet: report n_steps, n_a, n_d, relaxation factor, sparsity coefficient
- Feature importance: unified 0-100 scale across ML and DL models
  - LogReg: |coefficients| rescaled
  - SVM: permutation importance
  - RF: Gini importance
  - XGBoost/LightGBM/CatBoost: gain-based importance
  - MLP: permutation importance
  - TabNet: built-in attention masks
- Model comparison: frame as convergent findings, not horse race
- Stacking ensemble: report base learners, meta-learner, CV strategy
- Tree-based with small N: frame as "exploratory"; never claim generalizability

## Cross-Skill Interface

```
Method Unit Contract:
├── code_python      -> .py script (style-varied)
├── methods_md       -> methods.md (varied structure)
├── results_md       -> results.md (varied phrasing)
├── tables/          -> Markdown + CSV
├── figures/         -> PNGs 300 DPI (varied layout)
├── references_bib   -> .bib with cited references
└── comparison       -> cross-method narrative (in results.md)
```
