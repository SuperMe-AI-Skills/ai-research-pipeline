# 06 --- Fit Advanced Models (ViT + Ensemble)

## Executor: Main Agent (code generation)

## Data In: Code + prose from 05-analyze-subgroups.md (or 04 if no subgroup)

## Generate code for PART 5: Advanced Models

### 6A: ViT (Vision Transformer)
1. Load pre-trained ViT (vit_base_patch16_224) with ImageNet weights
2. Apply same data augmentation pipeline as CNN models
3. Architecture: ViT backbone -> [CLS] token -> Dropout -> [optional: concat tabular metadata] -> FC
4. Random search over hyperparameters (see config)
5. Early stopping on validation F1 (patience=2-3)
6. Evaluate: F1 + AUC with bootstrapped 95% CIs
7. Extract attention maps (mean over heads) for interpretability
8. Report: patch size, number of parameters, input resolution
9. Save: `plot_05_vit_confusion_roc.png`, `plot_05_vit_attention_maps.png`

### 6B: Ensemble
1. Soft voting ensemble: combine predicted probabilities from top CNN models + ViT
2. Stacking ensemble: train logistic regression meta-learner on base model predictions
3. Evaluate both: F1 + AUC with bootstrapped 95% CIs
4. Select best ensemble method by validation F1
5. Save: `plot_05_ensemble_confusion_roc.png`

### 6C: Tabular Metadata Handling
If extra features available (e.g., patient age, scanner type, image metadata):
- CNN models: concatenate at FC layer after global average pooling
- ViT: concatenate at [CLS] token representation
- If categorical metadata (device_id, site_id): use categorical embeddings

### 6D: Medical Imaging Specifics
If medical imaging dataset:
- DenseNet121: preferred architecture for medical imaging (report rationale)
- DICOM preprocessing: windowing, HU conversion, slice normalization
- Report any domain-specific preprocessing (e.g., lung windowing for CT)

### Quality: Code style variation
Apply per `reference/specs/code-style-variation.md`:
- Pick variable naming pattern (A-E)
- Pick comment style (A-E)
- Pick matplotlib style (A-D)
- Pick color palette (A-E)
- Randomize import order
- Record style vector for consistency

## Validation Checkpoint

- [ ] ViT fine-tuned with random search, best params reported
- [ ] Attention maps extracted and visualized
- [ ] Ensemble (soft voting + stacking) trained and evaluated
- [ ] F1 + AUC with 95% CIs for ViT and Ensemble
- [ ] Confusion matrices and ROC curves for ViT and Ensemble
- [ ] Parameter counts and FLOPs reported for ViT
- [ ] Tabular metadata integrated where available
- [ ] Code style variation applied

## Data Out -> 07-compare-models.md

```
advanced_code_py: [PART 5 Python code]
methods_para_vit: [prose]
methods_para_ensemble: [prose]
results_para_vit: [prose]
results_para_ensemble: [prose]
tables: [per-model metrics tables]
plots: [confusion/ROC per model, attention maps]
style_vector: [e.g., "B-A-C-D-E-2-1"]
```
