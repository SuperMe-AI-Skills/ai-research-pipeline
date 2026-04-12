# Experiment Standards

Hard rules for experiment quality in AI/ML methodology research.

## Code Structure Rules

1. **Random seed**: Set ONCE at the top. Record the seed value. Set for ALL sources of randomness.
   ```python
   import random, numpy as np, torch
   def set_seed(seed: int = 42):
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
   ```

2. **Package versions**: Document in requirements.txt with pinned versions.
   ```
   torch==2.3.0
   transformers==4.41.0
   numpy==1.26.4
   ```

3. **GPU reproducibility**: Enable deterministic mode and document hardware.
   ```python
   torch.use_deterministic_algorithms(True)
   # Record: GPU model, CUDA version, cuDNN version
   ```

4. **Output format**: Structured, self-describing results.
   ```python
   results = {
       "method": "proposed",
       "dataset": "cifar100",
       "seed": 42,
       "metrics": {"accuracy": 0.823, "f1_macro": 0.814},
       "params": 11_200_000,
       "flops": 1_800_000_000,
       "train_time_hours": 2.3,
       "config": {...}
   }
   json.dump(results, open("results/run_proposed_cifar100_seed42.json", "w"))
   ```

## Reporting Rules

| Element | Rule | Example |
|---------|------|---------|
| Multi-seed stats | ALWAYS report mean +/- std | "Accuracy: 82.3 +/- 0.4" |
| Bold best | Bold the best result per metric per dataset | **82.3** +/- 0.4 |
| Significance | Report p-value for key comparisons | "p = 0.003, paired bootstrap" |
| Parameters | Report total trainable params | "11.2M params" |
| FLOPs | Report per-sample forward FLOPs | "1.8 GFLOPs" |
| Wall-clock time | Report total training time | "4.2 GPU-hours on A100" |
| Inference speed | Report throughput or latency | "1200 samples/sec" |
| Datasets | Name, size, split, version | "CIFAR-100 (50K train, 10K test)" |

## Statistical Significance Testing

### Paired Bootstrap Test

For comparing two methods on the same test set:

```python
def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000, seed=42):
    """Two-sided paired bootstrap test.
    scores_a, scores_b: per-example metric values (same length).
    Returns p-value."""
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    delta_observed = np.mean(scores_a) - np.mean(scores_b)
    count = 0
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        delta_boot = np.mean(scores_a[idx]) - np.mean(scores_b[idx])
        if abs(delta_boot) >= abs(delta_observed):
            count += 1
    return count / n_bootstrap
```

### When to Use Which Test

| Scenario | Test | Details |
|----------|------|---------|
| Two methods, same test set | Paired bootstrap | 10,000 bootstrap samples |
| Two methods, multiple datasets | Wilcoxon signed-rank | Non-parametric paired test |
| Multiple methods, one dataset | McNemar's test (classification) | For classification only |
| Multiple methods, multiple datasets | Friedman + Nemenyi post-hoc | Rank-based comparison |

## Ablation Study Design

### Component Ablation Table Template

| Variant | Component Changed | Dataset 1 | Delta | Dataset 2 | Delta |
|---------|------------------|-----------|-------|-----------|-------|
| Full model | -- | XX.X | -- | XX.X | -- |
| w/o novel attention | Replace with standard | XX.X | -Y.Y | XX.X | -Y.Y |
| w/o auxiliary loss | Remove aux loss | XX.X | -Y.Y | XX.X | -Y.Y |
| w/o data augmentation | Remove augmentation | XX.X | -Y.Y | XX.X | -Y.Y |

### Ablation Rules

1. Change ONE component at a time (controlled experiment)
2. Use the SAME seeds as the full model
3. Report delta from full model, not absolute numbers alone
4. If removing component IMPROVES performance, investigate and report honestly
5. Include at least: the novel component, each loss term, key design choices

## Training Curve Standards

Every experiment must log:
- Training loss per step/epoch
- Validation metric per epoch
- Learning rate per step (if using scheduler)
- GPU memory usage (peak)
- Gradient norm (for diagnosing training stability)

### Training Curve Plot Requirements

- X-axis: epochs (or steps for large-scale)
- Y-axis: loss (left) and metric (right, if different scale)
- All methods on same plot for visual comparison
- Shaded region for std across seeds (if >= 3 seeds)
- Mark best validation epoch with vertical line

## Sanity Checks (Pre-Flight)

Run with debug config (2 epochs, small data) before full training:

| Check | What to Verify | Action if Failed |
|-------|----------------|------------------|
| No errors | Training completes 2 epochs | Fix bugs |
| Loss decreases | Loss at epoch 2 < epoch 1 | Check learning rate, gradient flow |
| Finite values | No NaN/Inf in loss or gradients | Check data, initialization, loss |
| Memory OK | Peak GPU memory < available | Reduce batch size, enable gradient checkpointing |
| Timing estimate | Extrapolate total runtime | Reduce scope if > MAX_TOTAL_GPU_HOURS |
| Checkpoint works | Save then resume produces same results | Fix checkpointing logic |

## Hyperparameter Sensitivity Analysis

### What to Vary

| Hyperparameter | Default Range | Method |
|---------------|---------------|--------|
| Learning rate | {1e-4, 3e-4, 1e-3, 3e-3} | Grid search |
| Batch size | {16, 32, 64, 128} | Grid search |
| Model size | {small, base, large} | Architecture variants |
| Dropout | {0.0, 0.1, 0.2, 0.3} | Grid search |
| Weight decay | {0.0, 0.01, 0.1} | Grid search |
| Warmup steps | {0, 100, 500, 1000} | Grid search |

### Reporting Format

Present as heatmap (2 parameters) or line plot (1 parameter) showing:
- Performance on validation set (not test)
- Range of values tested
- Selected value highlighted

## What NOT to Do

- Never cherry-pick seeds that favor your method
- Never omit a standard baseline
- Never report test results from only 1 seed
- Never report mean without standard deviation
- Never claim "state-of-the-art" without comparing to actual SOTA
- Never tune hyperparameters on test set
- Never use different compute budgets for proposed vs. baselines (unless the point is efficiency)
- Never hide negative results (datasets where your method underperforms)
- Never report training set performance as evidence of method quality
- Never claim causal relationships from correlational analysis
