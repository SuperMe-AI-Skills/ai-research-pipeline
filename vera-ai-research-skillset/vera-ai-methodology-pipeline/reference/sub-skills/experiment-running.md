# Run Experiment: AI/ML Training Deployment

Deploy and run AI/ML training experiments for: **$ARGUMENTS**

## Environment Detection

1. **Check local environment**:
   - Python: `which python3`, `python3 --version`
   - PyTorch: `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
   - GPU: `nvidia-smi` (GPU model, memory, utilization)
   - Available GPUs: `torch.cuda.device_count()`
   - JAX (optional): `python3 -c "import jax; print(jax.devices())"`
   - CUDA version: `nvcc --version`

2. **Check for remote compute** (if configured):
   - SSH config for GPU servers / clusters
   - SLURM / PBS scheduler availability

## Pre-flight Checks

Before launching any training run:

1. **Verify scripts are syntactically valid**:
   ```bash
   python3 -c "import py_compile; py_compile.compile('src/train.py')"
   ```

2. **Estimate runtime**:
   - Check epochs, batch size, dataset size, model parameters
   - Rough estimate: flag if > MAX_TOTAL_GPU_HOURS
   - Memory estimate: model params * 4 bytes * ~3 (optimizer states)

3. **Verify seed is set**: Every training run MUST have a reproducible seed

4. **Check disk space**: Checkpoints can be large (especially for large models)

5. **Verify data is accessible**: Dataset download complete, paths valid

## Deployment

### Local Single-GPU

```bash
# Standard training
nohup python3 src/train.py --config src/configs/default.yaml --seed 42 \
  --output_dir results/run_seed42 \
  > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > .train_pid
```

Use `run_in_background: true` for the Bash call.

### Multi-GPU (DataParallel / DDP)

```bash
# PyTorch DDP
nohup torchrun --nproc_per_node=NUM_GPUS src/train.py \
  --config src/configs/default.yaml --seed 42 \
  > logs/train_ddp_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Multi-Seed Runs

```bash
# Launch multiple seeds in sequence or parallel
for seed in 42 123 456; do
  nohup python3 src/train.py --config src/configs/default.yaml --seed $seed \
    --output_dir results/run_seed${seed} \
    > logs/train_seed${seed}.log 2>&1 &
done
```

### Remote Execution (via SSH)

1. **Sync code**: `rsync -avz --exclude='.git' --exclude='data/' ./ server:~/project/`
2. **Launch in screen/tmux**:
   ```bash
   ssh server "cd ~/project && screen -dmS train_main bash -c 'python3 src/train.py --config src/configs/default.yaml > logs/train.log 2>&1'"
   ```

### SLURM Cluster

```bash
sbatch --job-name=train_proposed --gres=gpu:1 --time=24:00:00 \
  --output=logs/slurm_%j.log train.sh
```

## Post-Launch Verification

After launching:

1. **Verify process is running**: Check PID or SLURM job
2. **Check initial output**: Read log after 60 seconds, verify first batch processed
3. **Verify GPU utilization**: `nvidia-smi` shows active use
4. **Estimate completion**: Based on first epoch timing
5. **Report to user**:
   ```
   Training launched:
   - Script: src/train.py
   - Config: src/configs/default.yaml
   - Model: ProposedModel (11.2M params)
   - Dataset: CIFAR-100 (50K train)
   - Epochs: 100
   - Estimated runtime: ~2 hours on 1x A100
   - Log: logs/train_20260405_103000.log
   - PID: 12345
   ```

## Key Rules

- ALWAYS set random seeds for ALL sources of randomness (random, numpy, torch, cuda)
- ALWAYS log output to a file (never just stdout)
- Save PID or job ID for monitoring
- Use JSON or YAML for results (not pickle for portability)
- Include per-epoch timing and GPU memory usage in logs
- Checkpoint every N epochs (not just at the end)
- For multi-seed: use the SAME config, only vary the seed
- Never launch training without estimating runtime and memory first
- If estimated time > MAX_TOTAL_GPU_HOURS, warn user and suggest alternatives
