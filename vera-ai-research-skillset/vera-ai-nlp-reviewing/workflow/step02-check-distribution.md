# 02 --- Check Distribution

## Executor: Main Agent (code generation)

## Data In: Structured input summary from 01-collect-inputs.md

## Generate code for PART 1

### Class Balance Check
- Frequency table of labels (count and proportion for each class)
- Proportion of minority class
- If minority class < 10%: print imbalance warning with implications
  (underrepresented class metrics unreliable, consider class_weight="balanced",
  SMOTE, or stratified sampling)

### Text Statistics
- Text length distribution (character count and word count)
  - Mean, median, min, max, std for both
- Vocabulary size (unique tokens after applying the text-cleaning rules
  defined in `03-run-primary-test.md` step "Data Preparation > Text cleaning":
  lowercase + remove URLs/handles/hashtags + strip punctuation + collapse
  whitespace)
- Top 20 most frequent tokens (excluding stop words)
- Text length by class (do classes have different length distributions?)

### Data Quality
- Missing values check (null/empty text entries)
- Duplicate text check (exact duplicates across rows)
- If group column: number of unique groups, distribution of samples per group

### Plots
- Class distribution bar chart with percentage labels
- Text length histogram (word count) with class overlay
- Save as `plot_01_data_overview.png` (12x5, 300 DPI)
  (consistent name across structured/NLP/image free skills)

### Decision logic (printed in console)

```
if minority_proportion >= 0.10:
    → "Class balance is adequate (minority = X.X%). Standard methods apply."
    → balance_flag = TRUE
else:
    → "WARNING: Class imbalance detected (minority = X.X%). Will use
       class_weight='balanced' and monitor per-class metrics."
    → balance_flag = FALSE
```

### Interpretation
Print 2 sentences: class balance assessment + implications for modeling approach.

## Validation Checkpoint

- [ ] Frequency table of labels printed (counts + proportions)
- [ ] Minority class proportion reported
- [ ] If minority < 10%, imbalance warning issued
- [ ] Text statistics complete (length, vocabulary, top tokens)
- [ ] Missing/duplicate check complete
- [ ] plot_01_data_overview.png generated
- [ ] balance_flag set (TRUE/FALSE)
- [ ] Decision statement printed

## Data Out -> 03-run-primary-test.md

```
balance_flag: TRUE | FALSE
minority_proportion: value
class_frequencies: {class_label: {n, pct}}
text_stats: {mean_words, median_words, vocab_size}
data_quality: {n_missing, n_duplicates, n_groups}
distribution_code_py: [PART 1 Python code block]
```
