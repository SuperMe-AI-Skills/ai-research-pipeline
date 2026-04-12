# Results Analyzing: Parse and Interpret ML Experiment Results

Analyze completed experiment results and produce structured analysis.

## Input

- Raw results files: JSON, CSV, or log files from training runs
- Multiple seeds per method
- Baseline results for comparison

## Procedure

### Step 1: Collect and Parse Results

For each method x dataset x seed combination:
```python
results = []
for method in methods:
    for dataset in datasets:
        for seed in seeds:
            path = f"results/run_{method}_{dataset}_seed{seed}/metrics.json"
            r = json.load(open(path))
            results.append({
                "method": method, "dataset": dataset, "seed": seed,
                **r  # metrics dict
            })
df = pd.DataFrame(results)
```

### Step 2: Aggregate Across Seeds

```python
summary = df.groupby(["method", "dataset"]).agg(
    mean=("metric", "mean"),
    std=("metric", "std"),
    n_seeds=("seed", "count")
)
```

### Step 3: Statistical Significance Testing

For each (proposed method, baseline) pair on each dataset:

```python
from scipy import stats

# Paired bootstrap test
proposed_scores = per_example_scores["proposed"]
baseline_scores = per_example_scores["baseline"]
p_value = paired_bootstrap_test(proposed_scores, baseline_scores)

# Or Wilcoxon signed-rank if only aggregate metrics per seed
proposed_seeds = [results per seed for proposed]
baseline_seeds = [results per seed for baseline]
stat, p = stats.wilcoxon(proposed_seeds, baseline_seeds)
```

### Step 4: Ablation Analysis

For each ablation variant:
```python
full_model_metric = mean_metric("full")
for variant in ablation_variants:
    variant_metric = mean_metric(variant)
    delta = variant_metric - full_model_metric
    # Report: variant name, metric, delta, significance
```

### Step 5: Generate Structured Output

Write analysis to `RESULTS_ANALYSIS.md`:

```markdown
# Experiment Results Analysis

## Summary
- Methods compared: {N}
- Datasets: {list}
- Seeds per method: {N}
- Total training runs: {N}

## Main Results Table
| Method | Dataset 1 | Dataset 2 | ... | Avg | Params | FLOPs |
|--------|-----------|-----------|-----|-----|--------|-------|
[mean +/- std, bold best, underline second]

## Statistical Significance
| Comparison | Dataset | p-value | Significant (alpha=0.05)? |
|-----------|---------|---------|--------------------------|

## Ablation Results
| Variant | Dataset 1 | Delta | Dataset 2 | Delta |
|---------|-----------|-------|-----------|-------|
[Full model first, then each ablation]

## Efficiency Comparison
| Method | Params | FLOPs | Train Time | Inference Speed |
|--------|--------|-------|------------|-----------------|

## Key Findings
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

## Concerns / Anomalies
- [Any unexpected results, high variance, failure cases]
```

## Mandatory Checks

| Check | Rule | Action if Failed |
|-------|------|------------------|
| No NaN/Inf | All metrics finite | Investigate training run |
| Sufficient seeds | >= 3 seeds per method | Run additional seeds |
| Significance tested | Key comparisons have p-values | Run bootstrap test |
| Ablation coherent | Removing components hurts | Re-examine claims |
| Variance reasonable | Std < 20% of mean | Check for seed sensitivity |
| Efficiency reported | Params + FLOPs for all methods | Measure and add |

## Key Rules

- NEVER report results from only 1 seed
- ALWAYS bold best results in tables
- ALWAYS report standard deviation alongside mean
- If proposed method loses on some datasets, report honestly
- If ablation shows a component doesn't help, flag for discussion
- Include per-seed results in appendix for full transparency
