# 06 --- Fit Advanced (Deep Learning) Models

## Executor: Main Agent (code generation)

## Data In: Code + prose from 05-analyze-subgroups.md (or 04 if no subgroup)

## Generate code for PART 5: Deep Learning Models

### 6A: GRU (Gated Recurrent Unit)
1. Build vocabulary from training texts (top 20000 words)
2. Pad sequences to max_len (determined from data, cap at 256)
3. Architecture: Embedding → Bidirectional GRU → Dropout → [optional: concat extra] → FC
4. Random search over hyperparameters (see config)
5. Early stopping on validation F1 (patience=2)
6. Evaluate: F1 + AUC with bootstrapped 95% CIs
7. Save: `plot_05_gru_confusion_roc.png`

### 6B: TextCNN
1. Same vocabulary and padding as GRU
2. Architecture: Embedding → Conv2d (filter sizes 3,4,5) → Max pool → [optional: concat extra] → FC
3. Random search over hyperparameters
4. Early stopping on validation F1
5. Evaluate: F1 + AUC with bootstrapped 95% CIs
6. Save: `plot_05_cnn_confusion_roc.png`

### 6C: ALBERT (Transformer)
1. Tokenize with AutoTokenizer (albert-base-v2)
2. Architecture: ALBERT → [CLS] → Dropout → [optional: concat tabular] → FC
3. Random search over hyperparameters
4. Early stopping on validation F1
5. Evaluate: F1 + AUC with bootstrapped 95% CIs
6. Save: `plot_05_albert_confusion_roc.png`

### 6D: Extra Features Handling
If extra features available (e.g., AI confidence, entropy, drug_id, cond_id):
- ML models: horizontally stack with TF-IDF (sparse)
- DL models: concatenate at FC layer
- If dict-mode extras (useful_z, drug_id, cond_id): use categorical embeddings

### Quality: Code style variation
Apply per `reference/specs/code-style-variation.md`:
- Pick variable naming pattern (A-E)
- Pick comment style (A-E)
- Pick matplotlib style (A-D)
- Pick color palette (A-E)
- Randomize import order
- Record style vector for consistency

## Validation Checkpoint

- [ ] GRU trained with random search, best params reported
- [ ] TextCNN trained with random search, best params reported
- [ ] ALBERT fine-tuned with random search, best params reported
- [ ] F1 + AUC with 95% CIs for all 3 DL models
- [ ] Confusion matrices and ROC curves for all 3
- [ ] Extra features integrated where available
- [ ] Code style variation applied

## Data Out -> 07-compare-models.md

```
dl_code_py: [PART 5 Python code]
methods_para_gru: [prose]
methods_para_cnn: [prose]
methods_para_albert: [prose]
results_para_gru: [prose]
results_para_cnn: [prose]
results_para_albert: [prose]
tables: [per-model metrics tables]
plots: [confusion/ROC per DL model]
style_vector: [e.g., "B-A-C-D-E-2-1"]
```
