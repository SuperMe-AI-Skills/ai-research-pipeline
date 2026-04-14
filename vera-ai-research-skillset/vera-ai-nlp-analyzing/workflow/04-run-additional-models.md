# 04 --- Run Additional ML Models

## Executor: Main Agent (code generation)

## Data In: PART 0-2 code from vera-ai-nlp-testing output

## Prerequisite: Testing workflow (PART 0-2) already executed.

## Generate code for PART 3: Additional ML Models

### For each ML model (SVM, Random Forest, LightGBM):

1. **Grid search** over model-specific hyperparameters (see config/default.json)
2. **Feature modes**: train both text-only and text+extra variants (if extra features available)
3. **Select best** by validation weighted F1
4. **Report best hyperparameters**
5. **Evaluate on test set**:
   - Classification report (precision, recall, F1 per class)
   - Bootstrapped F1 (1000 iterations, 95% CI)
   - Bootstrapped AUC (macro OVR, 1000 iterations, 95% CI)
   - Confusion matrix
   - Multi-class ROC curves
6. **Feature importance** (where applicable):
   - SVM: permutation importance (if RBF kernel)
   - RF: Gini importance + permutation importance
   - LightGBM: gain-based importance
   - All normalized 0-100 scale
7. **Save plots**: `plot_03_svm_confusion_roc.png`, `plot_03_rf_confusion_roc.png`, `plot_03_lgbm_confusion_roc.png`

### Quality: Apply sentence bank from `reference/patterns/sentence-bank.md`
- Vary whether F1 or AUC leads the interpretation sentence
- Vary per-class highlight selection (best class vs worst class first)
- Include 0-1 methodological justifications per model

## Validation Checkpoint

- [ ] All 3 ML models trained with grid search
- [ ] Best hyperparameters reported for each
- [ ] F1 + AUC with 95% CIs for each model
- [ ] Classification reports for each model
- [ ] Confusion matrices and ROC curves plotted
- [ ] Feature importance computed where applicable
- [ ] Sentence bank applied (no repeated phrasing patterns)

## Data Out -> 05-analyze-subgroups.md

```
additional_models_code_py: [PART 3 Python code]
methods_para_ml: [methods paragraph prose per model]
results_para_ml: [results paragraph prose per model]
plots: [list of new plot filenames]
tables: [list of new table data]
importance_tables: {svm: df, rf: df, lgbm: df}
```
