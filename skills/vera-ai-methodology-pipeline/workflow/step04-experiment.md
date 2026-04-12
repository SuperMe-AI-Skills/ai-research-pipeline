# Step 04: Run Experiments

> **Executor**: Main Agent (invokes sub-skills)
> **Input**: `src/`, `baselines/`, `data/` + `PIPELINE_STATE.json`
> **Output**: `results/` directory + `RESULTS_ANALYSIS.md`

---

## Execution Instructions

### 4.1 Deploy Training Runs

```
Read and execute reference/sub-skills/experiment-running.md
```

The experiment-running skill handles:
- Environment detection (local GPU vs. remote cluster)
- Pre-flight verification (seeds, packages, expected output structure)
- Deployment (direct, multi-GPU, or remote via screen/tmux)
- PID/job tracking for monitoring

**Experiment matrix**:
- Proposed method: full training with default hyperparameters
- All baselines: full training with their standard hyperparameters
- At least 3 random seeds per method for variance estimation
- Include timing per epoch and total wall-clock time

### 4.2 Monitor Progress

```
Read and execute reference/sub-skills/experiment-monitoring.md
```

Track running experiments:
- Check process status (alive/completed/failed)
- Read log files for progress (epoch, loss, metrics)
- Plot training curves (loss vs. epoch, metric vs. epoch)
- Estimate time remaining
- Report errors immediately (NaN loss, OOM, crashes)

If training appears stuck (no progress for > 30 minutes):
- Check for gradient issues (vanishing/exploding)
- Check for memory leaks
- Check for data loading bottlenecks
- Report to pipeline log

### 4.3 Ablation Studies

After main experiments complete, run ablation studies:

| Ablation Type | What to Vary | Purpose |
|---------------|-------------|---------|
| Component ablation | Remove/replace key components one at a time | Isolate contribution |
| Hyperparameter sensitivity | Vary learning rate, batch size, model size | Robustness check |
| Data efficiency | Train on 10%, 25%, 50%, 100% of data | Scaling behavior |
| Architecture variants | Vary depth, width, attention heads | Design choice validation |

### 4.4 Analyze Results

After all experiments complete:

```
Read and execute reference/sub-skills/results-analyzing.md
```

The results-analyzing skill produces:
- Structured results tables with mean +/- std across seeds
- Statistical significance tests (paired bootstrap or Wilcoxon signed-rank)
- Ablation tables with delta from full model
- Training curves comparison plots
- Convergence analysis

**Mandatory checks**:

| Check | Rule | Action if Failed |
|-------|------|------------------|
| No NaN/Inf | All metrics finite | Investigate training |
| Seed variance | Std across seeds reasonable | Add more seeds |
| Significance | Key comparisons have p-values | Run bootstrap test |
| Ablation coherent | Removing components hurts performance | Re-examine contribution claim |
| Efficiency reported | FLOPs, params, wall-clock time | Measure and add |

### 4.5 Generate Results Summary

Write `RESULTS_ANALYSIS.md`:

```markdown
# Experiment Results Analysis

## Setup
- Proposed method: {name}
- Baselines: {list}
- Datasets: {list}
- Seeds: {list}
- Hardware: {GPU type, count}

## Main Results

### Table 1: Primary Metrics (mean +/- std over seeds)
| Method | Dataset 1 | Dataset 2 | Dataset 3 | Params | FLOPs |
|--------|-----------|-----------|-----------|--------|-------|
| Proposed | XX.X +/- Y.Y | ... | ... | XM | XG |
| Baseline 1 | ... | ... | ... | ... | ... |
| Baseline 2 | ... | ... | ... | ... | ... |

### Table 2: Ablation Study
| Variant | Dataset 1 | Delta | Dataset 2 | Delta |
|---------|-----------|-------|-----------|-------|
| Full model | XX.X | -- | ... | -- |
| w/o component A | XX.X | -Y.Y | ... | -Y.Y |
| w/o component B | XX.X | -Y.Y | ... | -Y.Y |

### Table 3: Hyperparameter Sensitivity
| Learning Rate | Batch Size | Dataset 1 | Dataset 2 |
|---------------|-----------|-----------|-----------|
| 1e-3 | 32 | ... | ... |
| 3e-4 | 64 | ... | ... |

### Table 4: Statistical Significance
| Comparison | Test | p-value | Significant? |
|-----------|------|---------|--------------|
| Proposed vs Baseline 1 | Paired bootstrap | 0.003 | Yes |
| Proposed vs Baseline 2 | Paired bootstrap | 0.021 | Yes |

## Key Findings
1. {Finding 1 -- how proposed method compares overall}
2. {Finding 2 -- ablation insights}
3. {Finding 3 -- efficiency/scaling behavior}

## Potential Concerns
- {Any anomalies, unexpected results, failure cases}

## Figures to Generate
- Training curves (loss + metric vs. epoch, all methods)
- Performance vs. dataset size (data efficiency)
- Ablation bar chart
- Hyperparameter sensitivity heatmap
- Inference speed comparison
```

### 4.6 Update State

```json
{
  "stage": 4,
  "status": "completed",
  "experiments": {
    "n_methods": 4,
    "n_datasets": 3,
    "n_seeds": 3,
    "total_runs": 36,
    "total_gpu_hours": 48.2
  },
  "key_findings": [
    "Proposed method outperforms all baselines on 2/3 datasets",
    "Component A contributes +2.3% on average (ablation)"
  ],
  "timestamp": "..."
}
```

---

## Validation Checkpoints

| ID | Check Item | Pass Criteria | Failure Handling |
|----|------------|---------------|------------------|
| 4a | Training completed | All runs finished successfully | Diagnose, retry (3x) |
| 4b | Results files exist | Metrics saved per run in results/ | Check output path |
| 4c | No NaN/Inf | All metric values finite | Investigate training |
| 4d | Multi-seed variance | >= 3 seeds per method | Add more seeds |
| 4e | Significance tested | Key comparisons have p-values | Run bootstrap test |
| 4f | Ablation completed | At least 2 ablation variants run | Run missing ablations |
| 4g | RESULTS_ANALYSIS.md written | Non-empty, all tables present | Regenerate from raw results |

---

## Next Step
-> Step 05: External Review via Codex MCP
