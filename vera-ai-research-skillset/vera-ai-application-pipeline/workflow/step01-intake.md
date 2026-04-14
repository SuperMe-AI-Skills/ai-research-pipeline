# Step 01: Intake -- Research Question & Data Inspection

> **Executor**: Main Agent
> **Input**: User's research question + data file path
> **Output**: `PIPELINE_STATE.json` with structured metadata + data profile

---

## Execution Instructions

### 1.1 Collect Research Context

Ask the user (or extract from $ARGUMENTS):

| Field | Required | Example |
|-------|----------|---------|
| Research question | Yes | "Can transformer models predict protein stability from sequence?" |
| Hypotheses | Recommended | "Pre-trained protein LMs outperform CNN baselines on stability prediction" |
| Target discipline | Yes | NLP, Computer Vision, Bioinformatics, etc. |
| Target venue/style | Optional | NeurIPS, ICML, ACL, EMNLP, CVPR, etc. |
| Data modality | Optional | Text, images, tabular, multimodal |

If the user provided a clear research question in $ARGUMENTS, do not re-ask -- extract and confirm.

### 1.2 Load and Inspect Data

1. Identify the data file path from user input
2. Determine format: CSV, Parquet, JSON/JSONL, HDF5, image directory, audio directory, TFRecord
3. Load using appropriate method:
   - CSV/TSV: `pandas.read_csv()`
   - Parquet: `pandas.read_parquet()`
   - JSON/JSONL: `pandas.read_json()` or line-by-line
   - HDF5: `h5py` or `pandas.read_hdf()`
   - Images: scan directory structure (class folders or manifest file)
   - Audio: scan directory + load sample with librosa/torchaudio
   - HuggingFace dataset: `datasets.load_dataset()`

4. Generate data profile:

```
Data Profile:
+-- Dimensions: N samples x P features/columns
+-- Modality Signals:
|   +-- Text columns: [list with avg token length]
|   +-- Numeric columns: [list with ranges]
|   +-- Categorical columns: [list with level counts]
|   +-- Image paths: [list, sample resolution]
|   +-- Audio paths: [list, sample duration]
+-- Label/Target:
|   +-- Type: classification (N classes) / regression / sequence
|   +-- Distribution: [class counts or value range]
|   +-- Imbalance ratio: [majority/minority if classification]
+-- Missing Values:
|   +-- Total: X cells (Y%)
|   +-- Per column: [columns with >5% missing]
+-- Summary Statistics:
|   +-- Numeric: min, Q1, median, Q3, max, mean, SD
|   +-- Categorical: mode, n_levels, top 5 levels with counts
|   +-- Text: avg length, vocab size estimate, language
+-- Potential Issues:
    +-- Constant columns: [list]
    +-- Duplicate rows: [count]
    +-- Label leakage suspects: [columns highly correlated with target]
    +-- Class imbalance: [ratio]
```

### 1.3 Assign Variable Roles

Present the data profile and ask user to confirm:

1. **Target variable(s)**: Which column/field is the prediction target?
2. **Input features**: Which columns/fields are model inputs?
3. **Metadata columns**: Which columns are IDs, timestamps, or non-features?
4. **Subgroup variable**: Which column for stratified analysis? (optional)
5. **Exclusion criteria**: Any rows to exclude? (optional)
6. **Pre-existing splits**: Is there a train/val/test split column? (optional)

If the research question clearly implies roles (e.g., "predict Y from X"), pre-assign and confirm.

### 1.4 Write Initial State

Write `PIPELINE_STATE.json`:

```json
{
  "stage": 1,
  "status": "completed",
  "research_question": "...",
  "hypotheses": ["..."],
  "discipline": "...",
  "venue_style": "...",
  "data_file": "/path/to/data",
  "data_format": "csv",
  "n_samples": 5000,
  "n_features": 12,
  "modality_signals": {
    "text_columns": ["review_text"],
    "numeric_columns": ["rating", "length"],
    "categorical_columns": ["category"],
    "image_columns": [],
    "audio_columns": []
  },
  "variables": {
    "target": {"name": "sentiment", "type": "classification", "n_classes": 3, "missing_pct": 0.0},
    "inputs": [{"name": "review_text", "type": "text"}, {"name": "length", "type": "numeric"}],
    "metadata": [{"name": "id", "type": "identifier"}],
    "subgroup": null,
    "split_column": null
  },
  "exclusions": null,
  "timestamp": "..."
}
```

---

## Validation Checkpoints

| ID | Check Item | Pass Criteria | Failure Handling |
|----|------------|---------------|------------------|
| 1a | Data file exists | File found at path | Ask user to provide correct path |
| 1b | Data loads successfully | No read errors | Try alternative parser; report error |
| 1c | N > 0 | At least 1 sample after exclusions | Abort -- empty dataset |
| 1d | Target identified | target field populated | Ask user explicitly |
| 1e | At least 1 input feature | inputs array non-empty | Ask user explicitly |
| 1f | State file written | PIPELINE_STATE.json exists and valid JSON | Rewrite |

---

## Next Step
-> Step 02: Modality Detection & Routing
