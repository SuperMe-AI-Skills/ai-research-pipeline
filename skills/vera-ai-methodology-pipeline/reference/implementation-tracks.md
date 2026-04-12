# Implementation Tracks

Defines the three parallel implementation tracks for Stage 3. All tracks
launch simultaneously via the runtime-neutral `dispatch_track` operation
resolved in `workflow/step03-implement.md` section 3.2.0. The concrete
tool that backs `dispatch_track` depends on the runtime (`Agent`, `Task`,
`spawn_agent`, or a sequential fallback); this reference file stays
tool-agnostic.

## Track Architecture

```
Selected Idea
     |
     +-- Track A: Model Code -----> src/
     |     (always active)
     |
     +-- Track B: Baselines ------> baselines/
     |     (always active)
     |
     +-- Track C: Data Prep ------> data/ + src/data/
           (almost always active)
```

All tracks are **independent** -- no track depends on another's output. They can run fully in parallel.

## Track A: Model Code

**Always active.** Every AI/ML methodology contribution needs a reference implementation.

### Output Structure
```
src/
+-- models/
|   +-- proposed_model.py      # Main model architecture
|   +-- modules.py             # Reusable sub-modules (attention, pooling, etc.)
|   +-- __init__.py
|
+-- training/
|   +-- trainer.py             # Training loop (amp, grad accum, scheduling, checkpointing)
|   +-- loss.py                # Custom loss functions
|   +-- optimizer.py           # Optimizer + scheduler configuration
|   +-- __init__.py
|
+-- evaluation/
|   +-- evaluator.py           # Evaluation harness
|   +-- metrics.py             # Task-specific metrics
|   +-- __init__.py
|
+-- data/                      # (shared with Track C)
|   +-- data_loader.py         # Dataset classes + transforms
|   +-- __init__.py
|
+-- utils/
|   +-- logging.py             # W&B / TensorBoard logging
|   +-- checkpointing.py       # Save/resume logic
|   +-- reproducibility.py     # Seed setting, deterministic mode
|   +-- __init__.py
|
+-- configs/
|   +-- default.yaml           # Full experiment config
|   +-- debug.yaml             # Fast iteration config (small data, 2 epochs)
|
+-- train.py                   # Entry point: training
+-- evaluate.py                # Entry point: evaluation
+-- requirements.txt           # Pinned dependencies
```

### Model Implementation Requirements

| Element | Minimum | Notes |
|---------|---------|-------|
| Forward pass | Documented shapes | Input/output tensor shapes in docstring |
| Configuration | Via constructor or config | All hyperparameters externalized |
| Initialization | Explicit | Document weight init strategy |
| Mixed precision | Supported | torch.cuda.amp compatible |
| Gradient checkpointing | Optional | For memory-constrained settings |

### Training Pipeline Contract

Every training script must support:
```python
# Required CLI arguments / config fields
--seed INT              # Random seed (default: 42)
--epochs INT            # Number of training epochs
--batch_size INT        # Training batch size
--lr FLOAT              # Learning rate
--config PATH           # YAML config file
--checkpoint PATH       # Resume from checkpoint (optional)
--wandb / --no-wandb    # Enable/disable W&B logging
--output_dir PATH       # Where to save checkpoints + results
```

### Reproducibility Requirements

```python
import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Seed Counts

| Experiment Type | Seeds | Rationale |
|----------------|-------|-----------|
| Main results | >= 3 | Mean +/- std for tables |
| Ablation study | >= 3 | Same seeds as main for fair comparison |
| Hyperparameter search | 1 | Scout run, expand after selecting |
| Final best config | >= 5 | For robust reporting |

## Track B: Baselines

**Always active.** Fair comparison requires proper baseline implementations.

### Output Structure
```
baselines/
+-- baseline_standard.py      # Standard approach in the field
+-- baseline_recent.py         # Recent strong method from literature
+-- baseline_ablation.py       # Ablated version of proposed method
+-- shared_eval.py             # Common evaluation interface
+-- configs/
|   +-- baseline_standard.yaml
|   +-- baseline_recent.yaml
+-- README.md                  # Brief description of each baseline
```

### Baseline Selection Rules

| Priority | Type | Example |
|----------|------|---------|
| 1 (must) | Domain standard | ResNet for vision, BERT for NLP, MLP for tabular |
| 2 (must) | Recent SOTA | Best performing method from last 2 years |
| 3 (should) | Ablation | Proposed method minus the key novel component |
| 4 (nice) | Simple baseline | Linear probe, majority class, random |
| 5 (nice) | Concurrent work | Methods from simultaneous submissions |

### Evaluation Interface Contract

All methods (proposed + baselines) must expose:
```python
class ModelWrapper:
    def train(self, train_loader, val_loader, config) -> dict:
        """Train and return training metrics."""
        ...

    def evaluate(self, test_loader) -> dict:
        """Return {metric_name: value} dict."""
        ...

    def predict(self, inputs) -> Tensor:
        """Return predictions for inputs."""
        ...

    def count_parameters(self) -> int:
        """Return total trainable parameters."""
        ...

    def estimate_flops(self, input_shape) -> int:
        """Return estimated FLOPs for single forward pass."""
        ...
```

## Track C: Data Preparation

**Almost always active.** Only skip if using exclusively pre-existing data loaders (e.g., HuggingFace datasets with no custom processing).

### Output Structure
```
data/
+-- download_data.sh           # Automated download script
+-- README.md                  # Data sources, licenses, stats

src/data/
+-- data_loader.py             # Dataset classes
+-- transforms.py              # Data augmentation / preprocessing
+-- tokenizer.py               # Tokenization (NLP) or normalization (vision)
+-- data_stats.py              # Compute and print dataset statistics
```

### Common Benchmarks by Domain

| Domain | Standard Benchmarks | Where to Get |
|--------|-------------------|--------------|
| NLP - Classification | GLUE, SuperGLUE | HuggingFace datasets |
| NLP - Generation | WMT, CNN/DailyMail, XSum | HuggingFace datasets |
| NLP - QA | SQuAD, Natural Questions | HuggingFace datasets |
| Vision - Classification | ImageNet, CIFAR-10/100 | torchvision, TFDS |
| Vision - Detection | COCO, VOC | torchvision |
| Vision - Segmentation | ADE20K, Cityscapes | mmsegmentation |
| Tabular | UCI, OpenML, Kaggle | scikit-learn, openml |
| Audio | LibriSpeech, AudioSet | torchaudio |
| RL | Atari, MuJoCo, DMControl | gymnasium |
| Graphs | OGB, TU Datasets | PyG, DGL |

### Data Split Rules

1. Use official train/val/test splits when available
2. If no official split: stratified split with fixed seed
3. Never evaluate on training data
4. For small datasets: use k-fold cross-validation (k=5)
5. Document the split ratios and random seed

---

## Parallelization Rules

1. All active tracks launch simultaneously via `dispatch_track` (see
   `workflow/step03-implement.md` section 3.2.0) — no waiting between
   tracks when the bound `dispatch_mode` is `Agent`, `Task`, or
   `spawn_agent`. Under the `sequential` fallback, tracks run A → B → C
   in the main agent loop.
2. Each track is a separate isolated worker with its own context — the
   concrete isolation mechanism depends on `dispatch_mode` (subprocess,
   spawned agent, or main-loop subroutine for the sequential fallback).
3. Tracks share no intermediate outputs (fully independent)
4. The main agent monitors all tracks and collects their results (via
   tool-call responses, `wait_agent`, or direct returns depending on
   `dispatch_mode`)
5. Pre-flight check (Track A only) runs after Track A completes but
   doesn't block B or C (except under `sequential`, where A always
   completes before B starts anyway)
6. If any track fails: log error, continue other tracks, note gap in state
