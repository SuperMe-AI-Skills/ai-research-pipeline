# Experiment Monitoring: Track Training Progress

Monitor running AI/ML training experiments.

## What to Monitor

### Process Health

| Check | Command | Frequency |
|-------|---------|-----------|
| Process alive | `ps -p PID` or `squeue -j JOBID` | Every 5 min |
| GPU utilization | `nvidia-smi --query-gpu=utilization.gpu --format=csv` | Every 5 min |
| GPU memory | `nvidia-smi --query-gpu=memory.used --format=csv` | Every 5 min |
| Disk usage | `df -h` on output directory | Every 30 min |

### Training Progress

Read the latest entries from the log file:

| Metric | Where to Find | Alert Condition |
|--------|---------------|-----------------|
| Current epoch | Log file | No progress for > 30 min |
| Training loss | Log file or W&B | NaN/Inf, or increasing after epoch 10 |
| Validation metric | Log file or W&B | Decreasing for 10+ epochs (patience) |
| Learning rate | Log file | Stuck at 0 or unexpectedly high |
| Gradient norm | Log file (if logged) | > 100 (exploding) or < 1e-8 (vanishing) |
| Throughput | Log file | Dropped by > 50% from start |

### Checkpoint Management

| Check | Action |
|-------|--------|
| Latest checkpoint exists | Verify `checkpoints/` directory |
| Checkpoint loadable | Quick test: `torch.load(ckpt_path)` |
| Disk space sufficient | Alert if < 10 GB remaining |
| Old checkpoints cleaned | Keep top-K by validation metric |

## Monitoring Report Format

```markdown
## Training Status: {experiment_name}

**Time**: {timestamp}
**Elapsed**: {hours}h {minutes}m
**Progress**: Epoch {current}/{total} ({percent}%)

### Metrics
- Training loss: {latest} (trend: decreasing/stable/increasing)
- Validation metric: {latest} (best: {best} at epoch {best_epoch})
- Learning rate: {current}

### Resources
- GPU utilization: {percent}%
- GPU memory: {used}/{total} GB
- Disk used by checkpoints: {size} GB

### Status: HEALTHY / WARNING / ERROR
{Any alerts or concerns}

### ETA: {estimated_completion_time}
```

## Intervention Protocol

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| NaN loss | Learning rate too high, bad data batch, numerical instability | Kill run, reduce LR, add gradient clipping |
| OOM error | Batch size too large or model too big | Reduce batch size, enable gradient checkpointing |
| No GPU utilization | Data loading bottleneck or CPU fallback | Check data loader num_workers, verify CUDA |
| Loss plateau | LR too low, model capacity insufficient | Adjust scheduler, check if warmup completed |
| Validation diverging | Overfitting | Add regularization, early stopping |
| Slow throughput | I/O bottleneck | Pre-cache data, use faster storage |

## Key Rules

- Monitor at least once per 30 minutes during active training
- If training crashes, attempt automatic restart from latest checkpoint (up to 3 times)
- If loss is NaN, do NOT restart with same config -- diagnose first
- Log all monitoring observations to `logs/monitor.log`
- If estimated remaining time exceeds budget, alert user immediately
