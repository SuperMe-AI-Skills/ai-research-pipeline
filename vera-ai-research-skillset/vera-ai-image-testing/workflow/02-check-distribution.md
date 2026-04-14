# 02 --- Check Distribution

## Executor: Main Agent (code generation)

## Data In: Structured input summary from 01-collect-inputs.md

## Generate code for PART 1

### Class Balance Check
- Count images per class
- Proportion of minority class
- If minority class < 10%: print imbalance warning with implications
  (weighted loss, oversampling, class-aware augmentation)

### Image Statistics
- Image size distribution (height x width) if variable
- Channel statistics: mean and std per channel (R, G, B)
- Min/max pixel values
- Sample of 3 images per class for visual inspection

### Data Quality
- Corrupted/unreadable image check (attempt to load each, count failures)
- Duplicate image check (file hash or pixel-level comparison, sample-based)
- Image size outliers (unusually small or large images)

### Plots
- Subplot grid: (1) class distribution bar chart with counts,
  (2) sample image grid (3 images x N classes),
  (3) image size scatter (height vs width) if variable,
  (4) channel intensity histograms (R/G/B)
- Save as `plot_01_data_overview.png` (12x5, 300 DPI)

### Decision logic (printed in console)

```
if minority_proportion >= 0.10:
    → "Class balance is adequate (minority = X.X%). Standard methods apply."
    → balance_flag = TRUE
else:
    → "WARNING: Class imbalance detected (minority = X.X%). Will use
       weighted loss and augmentation for minority classes."
    → balance_flag = FALSE

if n_images >= 1000:
    → "Dataset size adequate for training CNN from scratch."
    → baseline = "cnn"
else:
    → "Small dataset (N = X). Will use pre-trained feature extractor."
    → baseline = "feature_extractor"
```

### Interpretation
Print 2 sentences: data quality assessment + recommended baseline approach.

## Validation Checkpoint

- [ ] Class distribution computed (counts + proportions)
- [ ] Minority class proportion reported
- [ ] If minority < 10%, imbalance warning issued
- [ ] Image statistics computed (size, channels, pixel range)
- [ ] Corrupted image check complete
- [ ] Sample images displayed per class
- [ ] plot_01_data_overview.png generated
- [ ] Baseline model decision made (CNN vs feature extractor)
- [ ] Decision statement printed

## Data Out -> 03-run-primary-test.md

```
balance_flag: TRUE | FALSE
minority_proportion: value
class_counts: {class_label: count}
image_stats: {mean_size, channels, pixel_range}
data_quality: {n_corrupted, n_duplicates}
baseline_model: "cnn" | "feature_extractor"
distribution_code_py: [PART 1 Python code block]
```
