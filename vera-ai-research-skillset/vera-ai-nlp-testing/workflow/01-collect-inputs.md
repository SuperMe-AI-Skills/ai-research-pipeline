# 01 --- Collect Inputs

## Executor: Main Agent

## Data In: User request (natural language)

## Required

Ask for anything missing. Do not proceed until all required inputs are collected.

1. **Data** --- one of:
   - CSV/data file uploaded
   - Dataset description (columns, types, N, source)
   - DataFrame already in memory

2. **Text column** --- the input text
   - Column name (e.g., "text", "review", "content")
   - Confirm contains free-text, not codes or IDs

3. **Label column** --- the target variable
   - Column name (e.g., "label", "sentiment", "category")
   - Number of classes (binary or multi-class)
   - Label meanings (e.g., 0 = human, 1 = AI-generated)

## Optional (collect for recommendation quality)

4. **Group column** --- for group-aware splitting
   - Prevents data leakage (e.g., same author in train and test)
5. **Extra features** --- numeric columns for augmentation
   - e.g., AI confidence scores, text metadata (word count, reading level)
6. **AI probability columns** --- for entropy/confidence feature engineering
   - Columns named ai_p_0, ai_p_1, ... (per-class probabilities from an AI detector)
7. **Sample size** --- if not evident from data

## Validation Checkpoint

- [ ] Text column identified and confirmed as free-text
- [ ] Label column identified with class count and meanings
- [ ] If N < 100, small-sample warning issued
- [ ] If classes > 10, high-cardinality warning issued
- [ ] If group column provided, confirmed for stratified splitting
- [ ] At least text column + label column collected

## Data Out -> 02-check-distribution.md

Structured input summary containing:
```
text_col: {name}
label_col: {name, n_classes, class_labels}
group_col: {name} or null
extra_features: [{name, type}] or null
ai_prob_cols: [col_names] or null
sample_size: N or "unknown"
data_source: {file_path | description | variable_list}
```
