# 01 --- Collect Inputs

## Executor: Main Agent

## Data In: User request (natural language)

## Required

Ask for anything missing. Do not proceed until all required inputs are collected.

1. **Data** --- one of:
   - CSV/data file uploaded
   - Dataset description (columns, types, N, source)
   - DataFrame already in memory

2. **Target column** --- the variable to predict
   - Column name (e.g., "diagnosis", "species", "price")
   - Confirm type: classification (categorical) or regression (continuous)

3. **Task type** --- classification or regression
   - If target has ≤ 20 unique values → likely classification
   - If target is continuous float → likely regression
   - Confirm with user if ambiguous

4. **Feature columns** --- predictors
   - Specific list, or "all columns except target"
   - Identify types: numeric, categorical, ordinal

## Optional (collect for recommendation quality)

5. **ID column** --- to exclude from features (e.g., patient_id, row_id)
6. **Group column** --- for group-aware splitting
7. **Domain context** --- for interpretation and recommendation
8. **Sample size** --- if not evident from data

## Validation Checkpoint

- [ ] Target column identified with type confirmed
- [ ] Task type confirmed (classification/regression)
- [ ] Feature columns identified with types
- [ ] If N < 50, small-sample warning issued
- [ ] If features > 500, high-dimensionality warning issued
- [ ] ID column excluded from features
- [ ] At least target + features collected

## Data Out -> 02-check-distribution.md

Structured input summary containing:
```
target_col: {name, type, n_classes_or_range}
task_type: "classification" | "regression"
feature_cols: [{name, type}]
id_col: {name} or null
group_col: {name} or null
sample_size: N or "unknown"
data_source: {file_path | description | variable_list}
```
