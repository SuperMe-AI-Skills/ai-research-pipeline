# Benchmark Patterns

Standard ML/DL benchmark protocols and comparison patterns for methodology research.

## Main Results Table Template

```latex
\begin{table}[t]
\caption{Main results on [benchmark suite]. Best results in \textbf{bold},
second best \underline{underlined}. Mean $\pm$ std over [N] seeds.}
\label{tab:main}
\centering
\begin{tabular}{lccc}
\toprule
Method & Dataset 1 & Dataset 2 & Dataset 3 \\
\midrule
\multicolumn{4}{l}{\textit{Baselines}} \\
Standard baseline & $X \pm Y$ & $X \pm Y$ & $X \pm Y$ \\
Recent SOTA & $X \pm Y$ & $X \pm Y$ & $X \pm Y$ \\
\midrule
\multicolumn{4}{l}{\textit{Ours}} \\
Proposed (full) & $\mathbf{X \pm Y}$ & $\mathbf{X \pm Y}$ & $\mathbf{X \pm Y}$ \\
\bottomrule
\end{tabular}
\end{table}
```

## Benchmark Protocol by Domain

### NLP Benchmarks

| Benchmark | Task | Metric | Split | Notes |
|-----------|------|--------|-------|-------|
| GLUE | Multi-task NLU | Avg score | Official dev/test | Report individual + avg |
| SuperGLUE | Multi-task NLU | Avg score | Official | Harder than GLUE |
| SQuAD v1.1/v2.0 | Extractive QA | EM / F1 | Official | v2.0 includes no-answer |
| WMT | Translation | BLEU / COMET | Official | Language pair matters |
| CNN/DailyMail | Summarization | ROUGE-1/2/L | Official | Also report BERTScore |
| MMLU | Knowledge | Accuracy | Official | 5-shot standard |

### Vision Benchmarks

| Benchmark | Task | Metric | Split | Notes |
|-----------|------|--------|-------|-------|
| ImageNet-1K | Classification | Top-1/5 Acc | Official | Also report params + FLOPs |
| CIFAR-10/100 | Classification | Accuracy | 50K/10K | Standard data augmentation |
| COCO | Detection | mAP@[.5:.95] | Official | Report AP_50 and AP_75 too |
| ADE20K | Segmentation | mIoU | Official | Single-scale vs multi-scale |
| STL-10 | Self-supervised | Linear probe acc | Official | Standard SSL eval |

### Tabular Benchmarks

| Benchmark | Task | Metric | Split | Notes |
|-----------|------|--------|-------|-------|
| UCI suite | Various | Accuracy/RMSE | 5-fold CV | Standard preprocessing |
| OpenML-CC18 | Classification | Accuracy | 10-fold CV | 72 datasets |
| Kaggle selected | Various | Task-specific | Train/test | Specify which competitions |

### Efficiency Benchmarks

| Metric | How to Measure | Report Format |
|--------|---------------|---------------|
| Parameters | `sum(p.numel() for p in model.parameters() if p.requires_grad)` | "11.2M" |
| FLOPs | Use fvcore or ptflops | "1.8 GFLOPs" |
| Throughput | Batched inference, warm up 10 iterations | "1200 img/sec on A100" |
| Latency | Single sample, median of 100 runs | "4.2ms on A100" |
| Memory | `torch.cuda.max_memory_allocated()` | "2.3 GB peak" |
| Training time | Wall-clock for full training | "4.2 GPU-hours" |

## Comparison Table Patterns

### Pattern 1: Method Comparison (Standard)

Best for: Comparing proposed method against baselines on standard benchmarks.

```
| Method | Params | FLOPs | Dataset 1 | Dataset 2 | Avg |
|--------|--------|-------|-----------|-----------|-----|
```

Rules:
- Group by method family (baselines above, ours below, separated by midrule)
- Bold best overall, underline second best
- Include params and FLOPs for efficiency context
- Report average across datasets in last column
- Cite source for baseline numbers (re-run vs. reported)

### Pattern 2: Ablation Study

Best for: Isolating contribution of each proposed component.

```
| # | Component A | Component B | Component C | Performance | Delta |
|---|-------------|-------------|-------------|-------------|-------|
| 1 | + | + | + | XX.X | -- (full) |
| 2 | - | + | + | XX.X | -Y.Y |
| 3 | + | - | + | XX.X | -Y.Y |
| 4 | + | + | - | XX.X | -Y.Y |
| 5 | - | - | - | XX.X | -Y.Y (base) |
```

Rules:
- Full model is row 1
- Remove ONE component per row
- Include "all removed" as bottom row to show cumulative gain
- Report delta from full model
- Use checkmarks (+/-) for clarity

### Pattern 3: Scaling Analysis

Best for: Showing how method scales with data, model size, or compute.

```
| Model Size | Params | Dataset Size | Performance | Train Time |
|-----------|--------|-------------|-------------|------------|
| Small | 5M | 10% | ... | ... |
| Small | 5M | 100% | ... | ... |
| Base | 50M | 10% | ... | ... |
| Base | 50M | 100% | ... | ... |
| Large | 200M | 100% | ... | ... |
```

Rules:
- Show performance vs. at least 2 axes (model size AND data size)
- Include compute cost for each configuration
- Plot as scaling curves (log-log if power law expected)

### Pattern 4: Transfer / Generalization

Best for: Showing method works beyond the training distribution.

```
| Pre-train | Fine-tune | Metric | Proposed | Baseline 1 | Baseline 2 |
|-----------|-----------|--------|----------|-----------|-----------|
| ImageNet | CIFAR-10 | Acc | ... | ... | ... |
| ImageNet | CIFAR-100 | Acc | ... | ... | ... |
| ImageNet | Flowers | Acc | ... | ... | ... |
```

## Statistical Significance Reporting

### In-Text Format

```
Our method achieves 82.3% accuracy (mean over 5 seeds), outperforming the
strongest baseline (80.1%) by 2.2 percentage points (paired bootstrap test,
p = 0.003, n = 10,000 bootstrap samples).
```

### In-Table Format

Use superscripts or footnotes:
```
| Method | Accuracy |
|--------|----------|
| Baseline | 80.1 +/- 0.3 |
| Proposed | **82.3 +/- 0.4*** |

* Significant improvement over Baseline (p < 0.01, paired bootstrap).
```

### When Significance Cannot Be Claimed

If p > 0.05, report honestly:
```
While our method achieves a higher mean accuracy (82.3 vs. 80.1), the
difference is not statistically significant (p = 0.12, paired bootstrap)
given the variance across seeds.
```

## Visualization Standards

### Training Curves

- Line plot: epoch on x-axis, metric on y-axis
- One line per method, distinguish by color AND line style
- Shaded region: mean +/- std across seeds
- Legend: method name + final metric value

### Ablation Bar Charts

- Grouped bar chart or horizontal bar
- Full model as reference line
- Color code: green for positive delta, red for negative
- Sort by impact (largest degradation first)

### Pareto Frontier Plots

- Scatter plot: x = efficiency metric (FLOPs, params), y = performance metric
- Pareto frontier line connecting non-dominated points
- Label each point with method name
- Use log scale for x-axis if range > 10x

### Attention / Feature Visualizations

- Show for a few representative examples (not cherry-picked)
- Include both success and failure cases
- Use colorbar with units/interpretation
- Overlay on original input for spatial methods

## Quality Checklist for Camera-Ready

| Check | Description |
|-------|-------------|
| Multi-seed | All results averaged over >= 3 seeds with std reported |
| Significance | Key comparisons have p-values |
| Baselines fair | Same compute budget, same data, same eval protocol |
| Ablation complete | Each novel component ablated individually |
| Efficiency reported | Params, FLOPs, wall-clock for all methods |
| Negative results | Failure cases discussed honestly |
| Reproducibility | Seeds, configs, hardware documented |
| Figure resolution | >= 300 DPI, vector PDF preferred |
| Table alignment | Decimal-aligned, bold best, consistent precision |
| Code release | Link to anonymous repo (for review) or public repo |
