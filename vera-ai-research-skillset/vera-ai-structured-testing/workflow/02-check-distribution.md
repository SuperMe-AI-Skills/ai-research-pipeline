# 02 --- Check Distribution

## Executor: Main Agent (code generation)

## Data In: Structured input summary from 01-collect-inputs.md

## Generate code for PART 1

### Missing Values Analysis
- Count and proportion of missing values per column
- If any column > 50% missing: flag for potential exclusion
- Overall missing rate across dataset
- Missing value heatmap or bar chart

### Feature Distributions
- For numeric features: mean, std, min, Q1, median, Q3, max, skewness
- For categorical features: number of levels, top 5 values with counts
- Outlier detection (IQR method): count and proportion per numeric feature

### Class Balance (classification only)
- Frequency table of target (count and proportion for each class)
- Proportion of minority class
- If minority class < 10%: print imbalance warning

### Correlation Analysis
- Pearson correlation matrix for numeric features
- Flag pairs with |r| > 0.90 (high multicollinearity)
- Target correlation: top 10 features most correlated with target

### Plots
- Subplot grid: (1) class balance bar chart, (2) correlation heatmap (top 15 features),
  (3) missing values bar chart, (4) target distribution
- Save as `plot_01_data_overview.png` (12x5, 300 DPI)

### Decision logic (printed in console)

```
if task_type == "classification":
    if minority_proportion >= 0.10:
        → "Class balance is adequate (minority = X.X%). Standard methods apply."
        → balance_flag = TRUE
    else:
        → "WARNING: Class imbalance detected (minority = X.X%). Will use
           class_weight='balanced' and monitor per-class metrics."
        → balance_flag = FALSE

if n_high_corr_pairs > 0:
    → "WARNING: X feature pairs have |r| > 0.90. Consider removing redundant features."

if n_missing_cols > 0:
    → "X columns have missing values. Will apply median/mode imputation."
```

### Interpretation
Print 2 sentences: data quality assessment + implications for modeling.

## Validation Checkpoint

- [ ] Missing values analyzed and reported
- [ ] Feature distributions computed
- [ ] Outlier detection complete
- [ ] Class balance checked (classification) or target distribution inspected (regression)
- [ ] Correlation matrix computed, high correlations flagged
- [ ] plot_01_data_overview.png generated
- [ ] Decision statements printed

## Data Out -> 03-run-primary-test.md

```
balance_flag: TRUE | FALSE (classification) or null (regression)
minority_proportion: value or null
data_quality: {n_missing_cols, n_outlier_cols, n_high_corr_pairs}
feature_summary: {n_numeric, n_categorical, n_total}
distribution_code_py: [PART 1 Python code block]
```
