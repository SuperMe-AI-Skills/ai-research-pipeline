# Step 03: Implementation (Parallel Tracks)

> **Executor**: Main Agent orchestrating parallel workers via `dispatch_track` (runtime-specific; see section 3.2.0 for the backend binding: `Agent` / `Task` / `spawn_agent` / sequential fallback).
> **Input**: `PIPELINE_STATE.json` (selected idea, environment)
> **Output**: `src/`, `baselines/`, `data/` directories populated

---

## Execution Instructions

Read `reference/implementation-tracks.md` for detailed track specifications.
Read `reference/experiment-standards.md` for training and evaluation requirements.
Read `reference/benchmark-patterns.md` for benchmark protocol patterns.

### 3.1 Determine Active Tracks

Based on the selected idea, determine which tracks to run:

| Track | Condition | Always? |
|-------|-----------|---------|
| A: Model Code | Idea involves a new architecture/method/loss | Yes (almost always) |
| B: Baselines | Need reference implementations for comparison | Yes (always) |
| C: Data Preparation | Idea requires dataset processing/creation | Yes (almost always) |

All active tracks launch via `dispatch_track` — the runtime-neutral
parallel-worker operation resolved in section 3.2.0 below.

### 3.2 Launch Parallel Tracks

#### 3.2.0 Resolve `dispatch_track` — the runtime-neutral parallel-worker operation

Before launching Tracks A/B/C, detect the runtime's parallel-worker
surface and bind `dispatch_track(prompt)` to a concrete tool call.
Check in priority order (see `SKILL.md` → Tool Usage → "Parallel-worker
surface" for full rationale):

1. **`Agent`** (Claude Code / Claude Agent SDK): `dispatch_track(prompt)`
   → emit an `Agent` tool call. Launch all three tracks by emitting
   multiple `Agent` calls in a single response; the runtime runs them
   in parallel.
2. **`Task`** (alternate Claude Code surface): same semantics as `Agent`.
3. **`spawn_agent` + `send_input` + `wait_agent`** (lifecycle-based
   runtimes): `dispatch_track(prompt)` →
   `id = spawn_agent(); send_input(id, prompt)`. Collect results via
   `wait_agent(id)` in section 3.3.
4. **Sequential fallback** (NO parallel-worker tool available):
   `dispatch_track(prompt)` → run the prompt yourself as a subroutine of
   the main agent loop, one track at a time (A → B → C). Record
   `"dispatch_mode": "sequential"` in `PIPELINE_STATE.json`. Wall-clock
   time increases but the output artifact set is identical.

Record the bound mode in `PIPELINE_STATE.json`:

```json
{"dispatch_mode": "Agent" | "Task" | "spawn_agent" | "sequential"}
```

The pipeline NEVER aborts because a specific parallel-worker tool is
missing — the sequential fallback is always available.

#### Track A: Model Code  *(dispatch via `dispatch_track`)*

```
Prompt: "Implement the proposed model/method for the following AI research idea:

Idea: {selected_idea.title}
Summary: {selected_idea.summary}
Hypothesis: {selected_idea.hypothesis}
Framework: {preferred_framework}

Create code in src/ with:

1. Model Architecture (src/models/):
   - proposed_model.py: Main model implementation
   - Clear module structure: __init__, forward, loss computation
   - Type hints and docstrings on all public methods
   - Configurable hyperparameters via constructor args or config dict

2. Training Pipeline (src/training/):
   - trainer.py: Training loop with:
     - Mixed precision support (torch.cuda.amp)
     - Gradient accumulation
     - Learning rate scheduling (warmup + cosine/linear decay)
     - Checkpointing (save/resume)
     - Logging (W&B or TensorBoard)
   - loss.py: Custom loss functions (if applicable)
   - optimizer.py: Optimizer configuration

3. Evaluation Pipeline (src/evaluation/):
   - evaluator.py: Evaluation metrics computation
   - metrics.py: Task-specific metrics (accuracy, F1, BLEU, FID, etc.)
   - Reproducible evaluation with fixed seeds

4. Configuration:
   - src/configs/default.yaml: Default hyperparameters
   - src/configs/debug.yaml: Small-scale config for fast iteration

5. Entry Points:
   - src/train.py: Main training script with argparse/hydra
   - src/evaluate.py: Standalone evaluation script
   - src/run_experiment.sh: Shell script for full experiment

Requirements:
- PyTorch (or JAX) with GPU support
- Deterministic mode: torch.manual_seed(), torch.backends.cudnn.deterministic
- All random seeds configurable
- Pre-flight check: run debug config for 2 epochs to verify no errors
- Document package dependencies in src/requirements.txt"
```

#### Track B: Baselines  *(dispatch via `dispatch_track`)*

```
Prompt: "Implement baseline methods for comparison with:

Idea: {selected_idea.title}
Method: {brief method description}

Create code in baselines/ with:

1. baseline_standard.py:
   - The standard/default approach in the field
   - Same evaluation interface as proposed method
   - Use established libraries where possible (HuggingFace, timm, etc.)

2. baseline_recent.py:
   - Recent strong baseline from literature survey
   - Re-implement or wrap existing open-source code
   - Same evaluation interface

3. baseline_ablation.py (if applicable):
   - Ablated version of proposed method (remove key component)
   - Helps isolate contribution of the novel component

4. Shared evaluation:
   - All baselines use the SAME evaluation metrics and test splits
   - Results saved in compatible format for comparison tables

Requirements:
- Same data loading pipeline as proposed method
- Same evaluation protocol (metrics, splits, seeds)
- Document which pretrained weights are used (if any)
- Include timing measurements per epoch/inference"
```

#### Track C: Data Preparation  *(dispatch via `dispatch_track`)*

```
Prompt: "Prepare datasets for:

Idea: {selected_idea.title}
Task type: {task description}

Create code in src/data/ and data/:

1. data_loader.py (src/data/):
   - Dataset class(es) with __getitem__ and __len__
   - Train/val/test splits (fixed random seed)
   - Data augmentation pipeline (if applicable)
   - Tokenization/preprocessing (if NLP)
   - Normalization/transforms (if vision)
   - Collate function for batching

2. download_data.sh or download_data.py (data/):
   - Automated download of benchmark datasets
   - Checksum verification
   - Standard splits from literature

3. data_stats.py (src/data/):
   - Dataset statistics (N, class distribution, sequence lengths, etc.)
   - Summary printed to stdout and saved to data/stats.json

Requirements:
- Use standard benchmark datasets from literature
- Document data sources, licenses, and preprocessing
- Support both local and streaming loading (for large datasets)
- Pin random seed for train/val/test split reproducibility
- Include a small synthetic dataset for unit testing"
```

### 3.3 Wait for All Tracks

Collect each dispatched worker's output per the `dispatch_mode` bound in
section 3.2.0:
- `Agent` / `Task`: the multi-call response returns each worker's result
  in the tool-result block.
- `spawn_agent`: call `wait_agent(id)` for each id spawned in 3.2 and
  collect its output.
- `sequential`: each `dispatch_track` call has already returned by the
  time the next one starts.

As each worker completes:
1. Verify output files exist
2. For Track A: run pre-flight check (debug config, 2 epochs)
3. For Track B: verify baselines run without errors
4. Log completion in PIPELINE_STATE.json

### 3.4 Pre-Flight Training Check

After Track A completes, run a quick verification:

```bash
python3 src/train.py --config src/configs/debug.yaml --epochs 2 --no-wandb
```

Verify:
- Code runs without errors on available hardware
- Loss decreases over 2 epochs (basic sanity)
- Output has expected structure (checkpoints, logs)
- No NaN/Inf in loss or gradients
- Memory usage is within GPU limits
- Estimated full training time is reasonable

If pre-flight fails: diagnose, fix code, re-run (up to 3 attempts).

### 3.5 Update State

```json
{
  "stage": 3,
  "status": "completed",
  "implementation_tracks": {
    "model_code": {"status": "completed", "preflight": "passed", "framework": "pytorch"},
    "baselines": {"status": "completed", "n_baselines": 3},
    "data_prep": {"status": "completed", "datasets": ["..."], "total_samples": 50000}
  },
  "estimated_training_hours": 12.5,
  "timestamp": "..."
}
```

---

## Validation Checkpoints

| ID | Check Item | Pass Criteria | Failure Handling |
|----|------------|---------------|------------------|
| 3a | Model code exists | `src/models/proposed_model.py` present | Regenerate |
| 3b | Pre-flight passes | Debug config runs 2 epochs without errors | Fix code, retry (3x) |
| 3c | Seeds set | Random seed in training script | Add seed |
| 3d | Baselines implemented | >= 2 baselines with same eval interface | Add standard baselines |
| 3e | Data pipeline works | Dataset loads and iterates without errors | Fix data loader |
| 3f | All code documented | Docstrings on public classes/functions | Add docstrings |
| 3g | Runtime reasonable | Estimated < MAX_TOTAL_GPU_HOURS | Reduce epochs or dataset size |

---

## Next Step
-> Step 04: Run Experiments
