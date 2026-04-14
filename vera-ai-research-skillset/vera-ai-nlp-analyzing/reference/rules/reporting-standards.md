# Reporting Standards --- Hard Rules

Apply to ALL generated output. Never violate.

1. **F1 score**: Always weighted F1 for multi-class. Report with 95% bootstrapped CI. Format: "F1 = 0.XXX, 95% CI [0.XXX, 0.XXX]".
2. **AUC**: Macro AUC (one-vs-rest) for multi-class. Report with 95% CI. Format: "AUC = 0.XXX, 95% CI [0.XXX, 0.XXX]".
3. **p-values**: Never "p = 0.000" -> use "p < .001". Exact to 3 decimals otherwise.
4. **Non-significance**: Never "no difference" -> "not statistically significant at alpha = [level]".
5. **Proportions**: Report as percentages with 1 decimal place (e.g., "38.2%").
6. **Sample sizes**: Report N_train, N_val, N_test. Report per-class counts.
7. **Hyperparameters**: Report best configuration found. Do not report full grid.
8. **Deep learning**: Report training epochs, best epoch, learning rate, batch size, early stopping patience.
9. **ALBERT**: Report pre-trained model name, max sequence length, whether base was frozen.
10. **Feature importance**: Unified 0-100 scale. Max feature = 100. Report top 20.
11. **Bootstrapped CIs**: 1000 iterations, 2.5th/97.5th percentiles. Always report.
12. **Model comparison**: Frame as convergent findings, NOT horse race. No "best model" declarations.
13. **Tree-based with small N**: Frame as "exploratory." Never claim generalizability without cross-validation.
14. **Decimal places**: 3 for F1/AUC/effect sizes, 3 for p-values, 1 for proportions.
15. **Confusion matrix**: Always report with counts. Per-class precision/recall in text.
16. **ROC curves**: Per-class curves with AUC values annotated.

## Visualization Standards

| Plot | When | Purpose |
|---|---|---|
| Class balance bar chart | Always | Label distribution |
| Confusion matrix heatmap | Per model | Classification performance |
| ROC curves (per-class) | Per model | Discrimination |
| Feature importance bar | After ML models | Feature ranking |
| Subgroup F1 bar chart | Subgroup analysis | Fairness/disparity |
| Model comparison bar | After all models | Performance overview |

All plots: 300 DPI, clean labels, no gridlines clutter.
Vary theme/palette per generation (see code-style-variation.md).
