# Figure Creating: Application Paper Visualizations

Generate publication-quality figures for applied AI/ML research papers.

## Input

- `PAPER_PLAN.md` -- figure plan
- `output/track_outputs/` -- results from each model track
- `output/figures/` -- any existing figures

## Standard Figure Set for Application Papers

### 1. Data Overview

- Class distribution bar chart
- Example inputs from each class/category
- Feature distribution plots (for structured data)
- Word cloud or length distribution (for NLP)
- Sample images (for vision)

### 2. Training Curves

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for method in methods:
    axes[0].plot(epochs, train_loss[method], label=method)
    axes[1].plot(epochs, val_metric[method], label=method)
axes[0].set_ylabel("Training Loss")
axes[1].set_ylabel("Validation Metric")
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.legend()
```

### 3. Model Comparison

- Grouped bar chart: models on x-axis, metric on y-axis, with error bars
- Radar/spider chart for multi-metric comparison (optional)
- Pareto plot: accuracy vs. efficiency

### 4. Error Analysis

- Confusion matrix heatmap (for classification)
- Per-class performance bar chart
- Error examples with model predictions
- Scatter plot of correct vs. incorrect predictions

### 5. Feature Importance / Interpretability

- SHAP summary plot (for structured/tabular)
- Attention heatmap (for NLP)
- Grad-CAM overlay (for vision)
- Permutation importance bar chart

## Style Requirements

| Property | Standard |
|----------|----------|
| Font size | >= 10pt |
| Colors | Colorblind-safe (tab10, Set2) |
| DPI | 300 PNG, vector PDF |
| Error bars | Mean +/- std with caps |
| Legend | Clear, not overlapping data |
| Axes | Always labeled with units |

## Output

For each figure:
- `paper/figures/fig_N.pdf` -- vector
- `paper/figures/fig_N.png` -- 300 DPI raster
- `paper/figures/gen_fig_N.py` -- generation script

## Key Rules

- Every figure must be reproducible (generation script)
- Colorblind-safe palettes only
- Include error bars from multi-seed runs
- Captions must be self-contained
- Include both success AND failure examples in qualitative figures
- Match figure style to target venue conventions
