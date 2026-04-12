# Reporting Standards --- Hard Rules

Apply to ALL generated output. Never violate.

## Classification Metrics
1. **F1 score**: Always weighted F1 for multi-class. Report with 95% bootstrapped CI. Format: "F1 = 0.XXX, 95% CI [0.XXX, 0.XXX]".
2. **AUC**: Macro AUC (one-vs-rest) for multi-class. Report with 95% CI. Format: "AUC = 0.XXX, 95% CI [0.XXX, 0.XXX]".
3. **Confusion matrix**: Always report with counts. Per-class precision/recall in text.
4. **ROC curves**: Per-class curves with AUC values annotated.

## Regression Metrics
5. **RMSE**: Root mean squared error. Report with 95% bootstrapped CI. Format: "RMSE = 0.XXX, 95% CI [0.XXX, 0.XXX]".
6. **R2**: Coefficient of determination. Report with 95% bootstrapped CI. Format: "R2 = 0.XXX, 95% CI [0.XXX, 0.XXX]".
7. **MAE**: Mean absolute error. Report with 95% bootstrapped CI. Format: "MAE = 0.XXX, 95% CI [0.XXX, 0.XXX]".
8. **Residual plots**: Always include predicted vs actual scatter and residual distribution histogram.

## General Rules
9. **p-values**: Never "p = 0.000" -> use "p < .001". Exact to 3 decimals otherwise.
10. **Non-significance**: Never "no difference" -> "not statistically significant at alpha = [level]".
11. **Proportions**: Report as percentages with 1 decimal place (e.g., "38.2%").
12. **Sample sizes**: Report N_train, N_val, N_test. Report per-class counts (classification) or distribution summary (regression).
13. **Hyperparameters**: Report best configuration found. Do not report full grid.
14. **Deep learning**: Report training epochs, best epoch, learning rate, batch size, early stopping patience.
15. **TabNet**: Report n_steps, n_d, n_a, relaxation factor (gamma), sparsity coefficient (lambda_sparse), scheduler type.
16. **MLP**: Report hidden layer sizes, activation, dropout rate, batch norm, weight decay.
17. **Stacking ensemble**: Report base learners used, meta-learner type, CV fold count, whether passthrough was enabled.
18. **CatBoost**: Report whether categorical features were handled natively; list which features were passed as cat_features.
19. **Feature importance**: Unified 0-100 scale. Max feature = 100. Report top 20.
   - LogReg: |coefficients| rescaled
   - SVM: permutation importance rescaled
   - RF: Gini importance rescaled
   - XGBoost: gain importance rescaled
   - LightGBM: gain importance rescaled
   - CatBoost: gain importance rescaled
   - TabNet: attention mask importance rescaled
20. **Bootstrapped CIs**: 1000 iterations, 2.5th/97.5th percentiles. Always report.
21. **Model comparison**: Frame as convergent findings, NOT horse race. No "best model" declarations.
22. **Tree-based with small N**: Frame as "exploratory." Never claim generalizability without cross-validation.
23. **Decimal places**: 3 for F1/AUC/RMSE/R2/MAE/effect sizes, 3 for p-values, 1 for proportions.
24. **Missing data**: Report missingness rates per feature. State imputation method and rationale.
25. **Encoding**: State encoding method per feature type. Note CatBoost native handling separately.

## Visualization Standards

| Plot | When | Purpose |
|---|---|---|
| Target distribution chart | Always | Label/outcome distribution |
| Confusion matrix heatmap | Per model (classification) | Classification performance |
| ROC curves (per-class) | Per model (classification) | Discrimination |
| Residual plot (pred vs actual) | Per model (regression) | Prediction accuracy |
| Residual histogram | Per model (regression) | Error distribution |
| Feature importance bar | After ML models | Feature ranking |
| TabNet attention heatmap | After TabNet | Per-instance feature selection |
| Subgroup metric bar chart | Subgroup analysis | Fairness/disparity |
| Model comparison bar | After all models | Performance overview |

All plots: 300 DPI, clean labels, no gridlines clutter.
Vary theme/palette per generation (see code-style-variation.md).
