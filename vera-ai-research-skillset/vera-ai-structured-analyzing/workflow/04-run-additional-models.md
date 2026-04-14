# 04 --- Run Additional ML Models

## Executor: Main Agent (code generation)

## Data In: PART 0-2 code from vera-ai-structured-testing output

## Prerequisite: Testing workflow (PART 0-2) already executed.

## Generate code for PART 3: Additional ML Models

### Task-Type Awareness

Detect task type from PART 0-2 output:
- **Classification**: use classifiers, report F1/AUC
- **Regression**: use regressors, report RMSE/R2/MAE

### For each ML model (SVM, Random Forest, XGBoost, LightGBM, CatBoost):

1. **Preprocessing pipeline**:
   - Numeric: impute missing (median default) + scale (StandardScaler)
   - Categorical (non-CatBoost): encode (OneHotEncoder for low cardinality, TargetEncoder for high cardinality)
   - CatBoost: pass categorical features natively via `cat_features` parameter
2. **Grid search** over model-specific hyperparameters (see config/default.json)
3. **Select best** by validation weighted F1 (classification) or validation RMSE (regression)
4. **Report best hyperparameters**
5. **Evaluate on test set**:
   - **Classification**:
     - Classification report (precision, recall, F1 per class)
     - Bootstrapped F1 (1000 iterations, 95% CI)
     - Bootstrapped AUC (macro OVR, 1000 iterations, 95% CI)
     - Confusion matrix
     - Multi-class ROC curves
   - **Regression**:
     - Bootstrapped RMSE (1000 iterations, 95% CI)
     - Bootstrapped R2 (1000 iterations, 95% CI)
     - Bootstrapped MAE (1000 iterations, 95% CI)
     - Residual plot (predicted vs actual)
     - Residual distribution histogram
6. **Feature importance**:
   - SVM: permutation importance (always, since structured data)
   - RF: Gini importance + permutation importance
   - XGBoost: gain-based importance
   - LightGBM: gain-based importance
   - CatBoost: gain-based importance (via `get_feature_importance()`)
   - All normalized 0-100 scale
7. **Save plots**: `plot_03_svm_results.png`, `plot_03_rf_results.png`, `plot_03_xgb_results.png`, `plot_03_lgbm_results.png`, `plot_03_catboost_results.png`

### Quality: Apply sentence bank from `reference/patterns/sentence-bank.md`
- Vary whether primary metric (F1/RMSE) or secondary metric (AUC/R2) leads the interpretation sentence
- Vary per-class highlight selection (best class vs worst class first) for classification
- Include 0-1 methodological justifications per model

## Validation Checkpoint

- [ ] All 5 ML models trained with grid search
- [ ] Best hyperparameters reported for each
- [ ] Appropriate metrics with 95% CIs for each model (F1+AUC or RMSE+R2+MAE)
- [ ] Classification reports or residual diagnostics for each model
- [ ] Confusion matrices/ROC curves (classification) or residual plots (regression)
- [ ] Feature importance computed for all models
- [ ] CatBoost used categorical features natively (no encoding applied)
- [ ] Sentence bank applied (no repeated phrasing patterns)

## Data Out -> 05-analyze-subgroups.md

```
additional_models_code_py: [PART 3 Python code]
methods_para_ml: [methods paragraph prose per model]
results_para_ml: [results paragraph prose per model]
plots: [list of new plot filenames]
tables: [list of new table data]
importance_tables: {svm: df, rf: df, xgb: df, lgbm: df, catboost: df}
```
