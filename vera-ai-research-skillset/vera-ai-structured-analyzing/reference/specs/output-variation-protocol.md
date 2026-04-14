# Output Quality Variation Protocol

Read this file before every generation. Apply all variation layers for
natural, diverse, non-repetitive output.

## Layer 1: Phrasing Variation (Sentence Bank)

For each ML/DL result, maintain 4-6 alternative phrasings. Select
contextually per generation. Never repeat the same phrasing pattern for
the same type of result within a single document.

See `reference/patterns/sentence-bank.md` for the full bank.

Rules:
- Rotate across paragraphs within the same document
- Select based on data context (model type, metric values, feature names)
- Someone reading 10 outputs should see 10 different interpretive choices

## Layer 2: Structure Variation

### Section ordering (within scientific validity constraints):

Choose ONE ordering for results.md based on research question:
- **Order A (benchmark-driven):** Baseline -> ML -> DL -> Stacking -> Comparison
- **Order B (complexity-driven):** Stacking -> TabNet -> MLP -> Gradient Boosting -> Linear -> Comparison
- **Order C (interpretability-driven):** Linear -> Importance -> Tree Ensemble -> DL -> Stacking -> Comparison

### Table and figure naming:
Vary: "Table 1. Model Performance Comparison" vs
"Table 1. Predictive Performance Across Methods" vs
"Table 1. Structured Data Classification/Regression Benchmarks"

### Figure layout:
Side-by-side vs stacked, grid vs individual --- vary across generations.

## Layer 3: Interpretation Depth Variation

Randomly include 1-2 of the following per analysis section:
- Practical significance framing ("the RMSE improvement of 0.03 corresponds to...")
- Comparison to published benchmarks ("consistent with gradient boosting results on similar tabular tasks")
- Limitation acknowledgment inline ("though the modest sample size limits...")
- Methodological justification ("CatBoost was included for its native categorical handling, avoiding information leakage from target encoding")

Generate contextually from actual data. These are NOT templates.

## Layer 4: Code Style Variation

See `reference/specs/code-style-variation.md` for the 7-dimension specification.

Apply per-generation variations to:
- Variable naming patterns
- Comment styles
- Section separators
- matplotlib/seaborn styles
- Color palettes
- Import order
- Function organization

## Layer 5: System Capabilities

What this system automates end-to-end:
- Correct model selection adapting to data characteristics
- Missing value and categorical handling that triggers appropriate strategies
- Cross-method comparison with nuanced interpretation
- Feature importance unification across disparate model types (coefficients, Gini, gain, permutation, attention)
- Consistent reporting across ML and DL paradigms
- Dual task support (classification and regression) with appropriate metrics
- Stacking ensemble with proper out-of-fold CV to prevent leakage
- CatBoost native categorical handling without encoding
- Citation accuracy and completeness
