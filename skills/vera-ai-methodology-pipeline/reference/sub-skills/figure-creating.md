# Figure Creating: Publication-Quality ML Visualizations

Generate publication-quality figures for AI/ML research papers.

## Input

- `PAPER_PLAN.md` -- figure plan with types and data sources
- `results/` -- raw experiment results
- `logs/` -- training logs

## Figure Types

### 1. Architecture Diagram

- Use matplotlib/tikz or draw.io for model architecture
- Show data flow, dimensions, key components
- Consistent notation with method section
- Clean, not cluttered -- omit obvious details

### 2. Training Curves

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Loss curves
for method, data in results.items():
    epochs = data["epochs"]
    mean_loss = np.mean(data["losses"], axis=0)  # over seeds
    std_loss = np.std(data["losses"], axis=0)
    ax1.plot(epochs, mean_loss, label=method)
    ax1.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss")
ax1.legend()

# Metric curves (similar for ax2)
```

### 3. Results Comparison (Bar Chart)

```python
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(datasets))
width = 0.2
for i, method in enumerate(methods):
    ax.bar(x + i*width, means[method], width, yerr=stds[method],
           label=method, capsize=3)
ax.set_xticks(x + width)
ax.set_xticklabels(datasets)
ax.set_ylabel("Accuracy (%)")
ax.legend()
```

### 4. Ablation Bar Chart

```python
fig, ax = plt.subplots(figsize=(8, 4))
variants = ["Full model", "w/o Comp A", "w/o Comp B", "w/o Comp C"]
values = [82.3, 79.1, 80.5, 81.0]
colors = ["#2ecc71" if v == max(values) else "#e74c3c" for v in values]
ax.barh(variants, values, color=colors)
ax.axvline(x=values[0], color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("Accuracy (%)")
```

### 5. Efficiency Pareto Plot

```python
fig, ax = plt.subplots(figsize=(8, 6))
for method, data in efficiency.items():
    ax.scatter(data["flops"], data["accuracy"], s=data["params"]/1e4,
               label=method, alpha=0.8)
ax.set_xlabel("GFLOPs")
ax.set_ylabel("Accuracy (%)")
ax.set_xscale("log")
ax.legend()
```

### 6. Attention / Feature Visualization

- Grad-CAM heatmaps overlaid on input
- Attention weight matrices
- t-SNE/UMAP of learned representations
- Include both success AND failure cases

## Style Requirements

| Property | Standard |
|----------|----------|
| Font | Matches LaTeX document font (use matplotlib rc) |
| Font size | >= 10pt for all text |
| Colors | Colorblind-safe: tab10, Set2, or custom palette |
| Line width | >= 1.5pt |
| DPI | 300 for PNG, vector for PDF |
| Figure size | Match column/page width of target venue |
| Legend | Inside plot or below, never overlapping data |
| Grid | Light gray, behind data |
| Axes labels | Always present, with units |
| Error bars | Mean +/- std, with caps |

## Output Format

For each figure, produce:
- `paper/figures/fig_N.pdf` -- vector format for LaTeX
- `paper/figures/fig_N.png` -- raster backup at 300 DPI
- `paper/figures/gen_fig_N.py` -- reproducible generation script

## LaTeX Inclusion

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig_N.pdf}
\caption{Description of the figure. Best viewed in color.}
\label{fig:name}
\end{figure}
```

## Key Rules

- Every figure must have a generation script (reproducibility)
- Use colorblind-safe palettes exclusively
- Include error bars/shaded regions from multi-seed runs
- Never use 3D plots when 2D suffices
- Captions should be self-contained (reader should understand without text)
- Include both success and failure examples in qualitative figures
