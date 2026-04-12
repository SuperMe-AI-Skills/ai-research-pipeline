# Method Track Definitions

Defines the parallel and sequential analysis tracks for each data modality.
Each track maps to specific workflow steps. The testing skill
provides workflow steps 01-03 (collect → diagnose → baseline) and the
analyzing skill provides workflow steps 04-08 (additional models → subgroups →
advanced/DL → comparison → manuscript).

## Track Architecture

```
Independent Tracks (parallel):
  T1: Data Diagnostics & Baseline   ← steps 01-03  (testing skill)
  T2: ML Model Battery              ← step 04      (analyzing skill)
  T3: Deep Learning Models          ← step 06      (analyzing skill)
  T4: Advanced / Ensemble           ← step 06      (ensemble portion)

Dependent Tracks (sequential):
  T5: Subgroup Analysis             ← step 05      (needs T1 results)

Post-Track (always sequential, after all tracks):
  Convergence: Model Comparison      ← step 07
  Assembly: Manuscript Fragments     ← step 08
```

The pipeline reads steps 01-03 from the `vera-ai-{modality}-testing` skill
and steps 04-08 from the `vera-ai-{modality}-analyzing` skill. No single
skill contains all eight files.

---

## NLP Text Classification

| Track | ID | Methods | Depends On |
|-------|----|---------|------------|
| Data Diagnostics & Baseline | T1_baseline | Text stats, class balance, TF-IDF + LogReg, F1/AUC with CIs | — |
| ML Model Battery | T2_ml | SVM (linear + RBF), Random Forest, LightGBM, grid search, feature importance | — |
| Deep Learning Models | T3_deep | GRU (bidirectional), TextCNN (multi-filter), ALBERT (fine-tuned), random search | — |
| Ensemble / Advanced | T4_ensemble | Weighted voting ensemble, stacking (if applicable), model selection | T2_ml, T3_deep |
| Subgroup Analysis | T5_subgroup | Per-subgroup evaluation, fairness metrics, performance disparity | T1_baseline |

### NLP-Specific Notes
- T3 supports optional tabular feature fusion (text + extra features)
- T3 DL models support dict-mode extras (useful_z, drug_id, cond_id)
- Feature importance only available for ML models (T2), not DL (T3)
- ALBERT requires GPU; fall back to GRU + TextCNN if GPU unavailable

---

## Structured / Tabular Data

| Track | ID | Methods | Depends On |
|-------|----|---------|------------|
| Data Diagnostics & Baseline | T1_baseline | Missing values, correlations, LightGBM baseline, feature importance | — |
| ML Model Battery | T2_ml | LogReg, SVM, RF, XGBoost, CatBoost, grid search | — |
| Deep Learning / Advanced | T3_deep | MLP, TabNet (if applicable), hyperparameter search | — |
| Ensemble | T4_ensemble | Stacking (meta-learner), weighted voting, blending | T2_ml, T3_deep |
| Subgroup Analysis | T5_subgroup | Per-subgroup evaluation, fairness analysis, feature interaction | T1_baseline |

### Structured-Specific Notes
- Supports both classification (F1/AUC) and regression (RMSE/R²) tasks
- CatBoost handles categorical features natively (no encoding needed)
- High-cardinality categoricals: target encoding or frequency encoding
- Missing value handling: median imputation (numeric), mode (categorical)

---

## Image Classification

| Track | ID | Methods | Depends On |
|-------|----|---------|------------|
| Data Diagnostics & Baseline | T1_baseline | Class balance, image stats, simple CNN or ResNet18 feature extractor | — |
| Transfer Learning Battery | T2_transfer | ResNet50, EfficientNet-B0, VGG16 (frozen + fine-tuned variants) | — |
| Advanced Architectures | T3_advanced | DenseNet121, Vision Transformer (ViT), advanced augmentation | — |
| Ensemble / Interpretability | T4_ensemble | Weighted voting ensemble, GradCAM visualization across models | T2_transfer, T3_advanced |
| Subgroup Analysis | T5_subgroup | Per-class performance, failure case analysis, domain-specific subgroups | T1_baseline |

### Image-Specific Notes
- All models use ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Two transfer learning strategies per model: feature extraction (frozen) + full fine-tuning
- GradCAM required for all CNN models in T4
- ViT requires images resized to 224x224
- Medical imaging: consider DenseNet121 as primary architecture

---

## Track Output Contract

Each track produces files in its directory `output/track_outputs/{track_id}/`:

```
{track_id}/
├── methods.md        ← Methods fragment for this track
├── results.md        ← Results fragment with metrics
├── code.py           ← Python code for this track
├── figures/          ← Track-specific figures (PNG, 300 DPI)
├── tables/           ← Track-specific tables (Markdown + CSV)
└── references.bib    ← Methodological references for this track
```

## Dependency Notation

`Depends On` cells use the **canonical full track IDs** from the same table
(e.g. `T1_baseline`, `T2_ml`, `T2_transfer`, `T3_deep`, `T3_advanced`).
Step 04 in the application pipeline matches these strings byte-for-byte
against `tracks_completed`, so short forms like `T1`, `T2`, or `T2, T3`
will silently strand dependent tracks.

- `—` = no dependencies, can run immediately
- `T1_baseline` = depends on the `T1_baseline` track completing first
- `T2_ml, T3_deep` = depends on both `T2_ml` AND `T3_deep` completing
