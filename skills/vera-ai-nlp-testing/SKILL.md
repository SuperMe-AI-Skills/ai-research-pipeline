---
name: vera-ai-nlp-testing
description: >-
  Runs class balance diagnostics and baseline text classification for NLP
  tasks. Produces class distribution tables, text length statistics,
  vocabulary analysis, TF-IDF feature inspection, a baseline Logistic
  Regression classifier with weighted F1 and macro AUC (bootstrapped 95%
  CIs), confusion matrix, and ROC curves. Ends with a recommendation
  block listing additional models and analyses available in the analysis
  workflow. Outputs Python scripts with 2 publication-quality plots. Triggered when
  user has text data and says "text classification," "NLP," "sentiment
  analysis," "AI detection," "language model detection," "document
  classification," "spam detection," "text mining," "topic classification,"
  or describes a task involving classifying text into categories.
  Does not handle structured/tabular-only data or image data.
user-invocable: true
allowed-tools: Read, Bash, Write, Edit
---

# NLP Text Classification --- Data Diagnostics & Baseline Modeling

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
1. CHECK CLASS BALANCE
   ├── Balanced (minority ≥ 10%) → standard methods
   └── Imbalanced (minority < 10%) → class weighting, note power limits

2. TEXT FEATURE STRATEGY
   ├── Vocab size < 50k → standard TF-IDF (max_features=10000)
   └── Vocab size ≥ 50k → sublinear TF-IDF with min_df=5

3. EXTRA FEATURES
   ├── Available → text + extra (augmented matrix)
   └── Not available → text-only (TF-IDF)
```

## Required Inputs

| Role | What to collect |
|---|---|
| **Text column** | Column name containing text data |
| **Label column** | Target variable, number of classes |
| **Group column** | For group-aware splitting (optional) |
| **Extra features** | Numeric columns for augmentation (optional) |

## Code Structure

```
PART 0: Setup & Data Loading
PART 1: Data Diagnostics          → plot_01_data_overview.png
PART 2: Baseline Classification   → plot_02_confusion_roc.png
PART 3: Recommendation Block      → text pointing to analysis workflow
```

## Reporting Standards

1. Metrics: weighted F1 and macro AUC (OVR) — always with 95% bootstrapped CIs
2. Format: "F1 = 0.XXX, 95% CI [0.XXX, 0.XXX]"
3. AUC: "AUC = 0.XXX, 95% CI [0.XXX, 0.XXX]"
4. Decimal places: 3 for F1/AUC/p-values, 1 for percentages, 0 for counts
5. Proportions: report as percentages with 1 decimal
6. Sample size: report final analytic N (train/val/test split sizes)
7. Non-significance: "not statistically significant at alpha = .05" — never "no difference"
8. Bootstrapped CIs: 1000 iterations, 2.5th/97.5th percentiles

## Models Available

| Workflow | Models |
|---|---|
| Testing (this skill) | Logistic Regression (TF-IDF baseline) |
| Analysis (vera-ai-nlp-analyzing) | SVM, Random Forest, LightGBM, GRU, TextCNN, ALBERT |

## Example Dataset

Any text classification dataset. Default: 20 Newsgroups (sklearn).
Python: `from sklearn.datasets import fetch_20newsgroups`

## Cross-Skill Interface

```
Output:
├── code_python      → .py script
├── methods_md       → methods.md baseline fragment
├── results_md       → results.md baseline results fragment
├── tables/          → Markdown/CSV tables for class balance + metrics
├── figures/         → 2 PNGs (class balance + confusion/ROC)
├── references_bib   → .bib with baseline/evaluation citations
└── recommendations  → text block (what analysis workflow produces)
```

When this skill is used as `T1_baseline` inside `vera-ai-application-pipeline`,
the standardized track artifacts above are REQUIRED, not optional.
