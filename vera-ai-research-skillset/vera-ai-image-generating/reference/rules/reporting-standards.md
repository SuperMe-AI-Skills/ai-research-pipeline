# Reporting Standards --- Hard Rules

Apply to ALL generated output. Never violate.

1. **F1 score**: Always weighted F1 for multi-class. Report with 95% bootstrapped CI. Format: "F1 = 0.XXX, 95% CI [0.XXX, 0.XXX]".
2. **AUC**: Macro AUC (one-vs-rest) for multi-class. Report with 95% CI. Format: "AUC = 0.XXX, 95% CI [0.XXX, 0.XXX]".
3. **p-values**: Never "p = 0.000" -> use "p < .001". Exact to 3 decimals otherwise.
4. **Non-significance**: Never "no difference" -> "not statistically significant at alpha = [level]".
5. **Proportions**: Report as percentages with 1 decimal place (e.g., "38.2%").
6. **Sample sizes**: Report N_train, N_val, N_test. Report per-class counts.
7. **Image dimensions**: Always report input resolution as HxWxC (e.g., 224x224x3). Report original image dimensions if resized.
8. **Data augmentation**: Report full augmentation pipeline with parameters (e.g., "RandomRotation(degrees=15)"). Specify train-only vs all splits.
9. **Transfer learning**: Report backbone name, pre-trained dataset (e.g., ImageNet-1K), frozen vs fine-tuned layers, differential learning rates if used.
10. **Parameter counts**: Report total parameters and trainable parameters per model. Report FLOPs.
11. **CNN models**: Report architecture variant (e.g., ResNet50, not just ResNet), pre-trained weights version, final classifier head dimensions.
12. **ViT**: Report pre-trained model name, patch size, input resolution, whether backbone was frozen, number of attention heads.
13. **GradCAM**: Report target layer per architecture. Show overlay visualizations with colorbar. Report top attributed regions.
14. **Attention maps**: Report head aggregation method (mean/max). Overlay on original images with transparency.
15. **Feature attribution**: Unified 0-100 scale. Max region = 100. Report top regions with consensus scores.
16. **Bootstrapped CIs**: 1000 iterations, 2.5th/97.5th percentiles. Always report.
17. **Model comparison**: Frame as convergent findings, NOT horse race. No "best model" declarations.
18. **CNN with small dataset**: Frame as "exploratory." Never claim generalizability without cross-validation.
19. **Decimal places**: 3 for F1/AUC/effect sizes, 3 for p-values, 1 for proportions.
20. **Confusion matrix**: Always report with counts. Per-class precision/recall in text.
21. **ROC curves**: Per-class curves with AUC values annotated.
22. **Medical imaging**: Report DICOM preprocessing (windowing, HU conversion), slice selection criteria, DenseNet rationale for medical tasks.
23. **Ensemble**: Report base model selection rationale, aggregation method (soft voting / stacking), meta-learner if applicable.

## Visualization Standards

| Plot | When | Purpose |
|---|---|---|
| Class balance bar chart | Always | Label distribution |
| Sample image grid | Always | Representative examples per class |
| Confusion matrix heatmap | Per model | Classification performance |
| ROC curves (per-class) | Per model | Discrimination |
| GradCAM overlay grid | After CNN models | Spatial attribution |
| ViT attention map grid | After ViT | Patch-level attention |
| Subgroup F1 bar chart | Subgroup analysis | Fairness/disparity |
| Model comparison bar | After all models | Performance overview (include params/FLOPs) |
| Training curves | Per model | Loss/F1 over epochs |

All plots: 300 DPI, clean labels, no gridlines clutter.
Vary theme/palette per generation (see code-style-variation.md).
