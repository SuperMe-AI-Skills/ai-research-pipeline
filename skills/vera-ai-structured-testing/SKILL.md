---
name: vera-ai-structured-testing
description: >-
  Runs data quality diagnostics and baseline classification/regression for
  structured (tabular) data. Produces missing value analysis, feature
  distributions, correlation matrix, class balance check, outlier detection,
  a baseline LightGBM classifier with weighted F1 and macro AUC (bootstrapped
  95% CIs), feature importance, confusion matrix, and ROC curves. Ends with
  a recommendation block listing additional models available in the analysis
  workflow. Outputs Python scripts with 2 publication-quality plots. Triggered when user
  has tabular/structured data and says "tabular data," "structured data,"
  "classification," "regression," "feature engineering," "predict from columns,"
  "CSV classification," "spreadsheet," "predict outcome," or describes a task
  involving predicting from numeric/categorical columns. Does not handle
  free-text NLP or image data.
user-invocable: true
allowed-tools: Read, Bash, Write, Edit
---

# Structured Data --- Data Quality Diagnostics & Baseline Modeling

Open-source skill.

## Workflow

Read each step file in `workflow/` before executing that step.

| Step | File | Executor | Output |
|---|---|---|---|
| Collect | `workflow/01-collect-inputs.md` | Main Agent | Structured input summary |
| Diagnose | `workflow/02-check-distribution.md` | Main Agent | PART 1 code block |
| Baseline | `workflow/03-run-primary-test.md` | Main Agent | PART 2-3 code blocks + T1 track artifacts |

## Decision Tree

```
1. TASK TYPE
   ├── Classification (target is categorical) → F1 + AUC metrics
   └── Regression (target is continuous) → RMSE + R² metrics

2. CHECK CLASS BALANCE (classification only)
   ├── Balanced (minority ≥ 10%) → standard methods
   └── Imbalanced (minority < 10%) → class weighting, note power limits

3. FEATURE TYPES
   ├── All numeric → standard scaling
   ├── Mixed (numeric + categorical) → encoding + scaling
   └── High cardinality categorical → target encoding or frequency encoding
```

## Required Inputs

| Role | What to collect |
|---|---|
| **Target column** | Column name, type (classification/regression) |
| **Feature columns** | List or "all except target" |
| **Task type** | Classification or regression |
| **ID column** | To exclude from features (optional) |

## Code Structure

```
PART 0: Setup & Data Loading
PART 1: Data Quality Diagnostics   → plot_01_data_overview.png
PART 2: Baseline Classification    → plot_02_confusion_roc.png
PART 3: Recommendation Block       → text pointing to analysis workflow
```

## Reporting Standards

1. Classification: weighted F1 and macro AUC (OVR) --- always with 95% bootstrapped CIs
2. Regression: RMSE and R² --- always with 95% bootstrapped CIs
3. Format: "F1 = 0.XXX, 95% CI [0.XXX, 0.XXX]"
4. Feature importance: top 20 features by gain, normalized 0-100
5. Decimal places: 3 for metrics, 1 for percentages, 0 for counts
6. Sample size: report final analytic N (train/val/test split sizes)
7. Non-significance: "not statistically significant at alpha = .05" --- never "no difference"

## Models Available

| Workflow | Models |
|---|---|
| Testing (this skill) | LightGBM (baseline) |
| Analysis (vera-ai-structured-analyzing) | LogReg, SVM, RF, XGBoost, CatBoost, MLP, Stacking Ensemble |

## Example Dataset

sklearn built-ins: `load_breast_cancer()` (binary), `load_iris()` (multi-class), `load_wine()` (multi-class).
Python: `from sklearn.datasets import load_breast_cancer`

## Cross-Skill Interface

```
Output:
├── code_python      → .py script
├── methods_md       → methods.md baseline fragment
├── results_md       → results.md baseline results fragment
├── tables/          → Markdown/CSV tables for diagnostics + metrics
├── figures/         → 2 PNGs (data overview + confusion/ROC)
├── references_bib   → .bib with baseline/evaluation citations
└── recommendations  → text block (what analysis workflow produces)
```

When this skill is used as `T1_baseline` inside `vera-ai-application-pipeline`,
the standardized track artifacts above are REQUIRED, not optional.
