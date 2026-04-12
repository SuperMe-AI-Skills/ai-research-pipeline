# Step 01: Research Direction Intake

> **Executor**: Main Agent
> **Input**: $ARGUMENTS (research direction) + local project files
> **Output**: `PIPELINE_STATE.json` with research context

---

## Execution Instructions

### 1.1 Parse Research Direction

Extract from $ARGUMENTS:
- **Research direction**: The broad AI/ML methodological area (e.g., "efficient fine-tuning methods for large language models with limited compute")
- **Specificity level**: Is this broad ("deep learning") or narrow ("parameter-efficient adaptation of vision transformers for few-shot medical image classification")?

If too broad (< 5 words), ask user to narrow:
```
Your direction "{direction}" is quite broad. Could you narrow it?
For example:
- "efficient attention mechanisms for long-context transformers"
- "self-supervised pre-training for tabular data"
- "robust training methods for noisy label learning in NLP"
```

### 1.2 Scan Existing Work

Check for prior work in the project directory:

| Directory | What to Look For |
|-----------|-----------------|
| `papers/`, `literature/`, `references/` | PDFs the user has already collected |
| `models/`, `architectures/` | Existing model code or architecture definitions |
| `experiments/`, `runs/`, `logs/` | Prior training runs or experiment results |
| `data/`, `datasets/` | Prepared datasets or data loaders |
| `results/`, `checkpoints/` | Prior model checkpoints or evaluation results |
| `paper/`, `manuscript/` | Draft manuscript in progress |
| `IDEA_DISCOVERY_REPORT.md` | Prior idea discovery run (enriched) |
| `IDEA_REPORT.md` | Prior raw idea brainstorm (from idea-creator) |
| `AUTO_REVIEW.md` | Prior review loop |
| `PIPELINE_STATE.json` | Prior pipeline state (check for resume) |

If `PIPELINE_STATE.json` exists with `status: "in_progress"` and is < 24 hours old:
- Offer to **resume** from last checkpoint
- Or **fresh start** (user chooses)

### 1.3 Identify Computational Environment

Detect available tools:
- Python: check for `python3 --version`
- PyTorch: check for `python3 -c "import torch; print(torch.__version__)"`
- GPU: check for `python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`
- JAX: check for `python3 -c "import jax; print(jax.__version__)"`
- LaTeX: check for `latexmk --version` or `pdflatex --version`
- Conda/venv: check for active environment
- Weights & Biases: check for `python3 -c "import wandb"`
- HuggingFace: check for `python3 -c "import transformers"`

### 1.4 Set Up Project Structure

Create directories if they don't exist (all at project root):
```
src/
src/models/
src/data/
src/training/
src/evaluation/
baselines/
experiments/
results/
logs/
data/
checkpoints/
paper/
paper/sections/
paper/figures/
```

**Note**: `baselines/` and `data/` are top-level directories, NOT nested under `src/`.
`logs/` is top-level (experiment-running writes `logs/train_*.log`, monitor-experiment tails them there).

### 1.5 Write Initial State

```json
{
  "stage": 1,
  "status": "completed",
  "research_direction": "...",
  "specificity": "narrow",
  "existing_work": {
    "papers": 5,
    "prior_models": false,
    "prior_experiments": false,
    "prior_manuscript": false
  },
  "environment": {
    "python_available": true,
    "pytorch_available": true,
    "gpu_available": true,
    "gpu_count": 1,
    "gpu_type": "...",
    "jax_available": false,
    "latex_available": true,
    "wandb_available": true,
    "preferred_framework": "pytorch"
  },
  "timestamp": "..."
}
```

---

## Validation Checkpoints

| ID | Check Item | Pass Criteria | Failure Handling |
|----|------------|---------------|------------------|
| 1a | Direction provided | Non-empty, >= 5 words | Ask user to elaborate |
| 1b | Python available | python3 found | Error -- need computation |
| 1c | Deep learning framework available | PyTorch or JAX found | Error -- need DL framework |
| 1d | Directory structure created | All dirs exist | Retry mkdir |
| 1e | State file written | PIPELINE_STATE.json valid JSON | Rewrite |

---

## Next Step
-> Step 02: Idea Discovery
