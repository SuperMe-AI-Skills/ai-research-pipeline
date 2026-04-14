# 06 --- Fit Advanced (Deep Learning) Models

## Executor: Main Agent (code generation)

## Data In: Code + prose from 05-analyze-subgroups.md (or 04 if no subgroup)

## Generate code for PART 5: Deep Learning Models

### 6A: MLP (Multi-Layer Perceptron)
1. Preprocess all features: impute, encode categoricals (one-hot or entity embeddings), scale numerics
2. Architecture: Input -> [BatchNorm] -> Dense(hidden1) -> ReLU/GELU -> Dropout -> Dense(hidden2) -> ReLU/GELU -> Dropout -> [optional: Dense(hidden3)] -> Output
3. Output layer: Softmax (classification) or Linear (regression)
4. Loss: CrossEntropyLoss (classification) or MSELoss (regression)
5. Random search over hyperparameters (see config)
6. Early stopping on validation F1 (classification) or validation RMSE (regression)
7. Evaluate:
   - **Classification**: F1 + AUC with bootstrapped 95% CIs
   - **Regression**: RMSE + R2 + MAE with bootstrapped 95% CIs
8. Save: `plot_05_mlp_results.png`

### 6B: TabNet
1. Preprocess: impute missing values, encode categoricals (label encoding for TabNet), scale numerics
2. Architecture: TabNet with sequential attention mechanism
   - n_d/n_a: decision/attention embedding dimensions
   - n_steps: number of sequential attention steps
   - gamma: coefficient for feature reusage
   - lambda_sparse: sparsity regularization
3. Random search over hyperparameters
4. Early stopping on validation metric (F1 or RMSE)
5. Evaluate:
   - **Classification**: F1 + AUC with bootstrapped 95% CIs
   - **Regression**: RMSE + R2 + MAE with bootstrapped 95% CIs
6. Extract TabNet feature importance from attention masks
7. Save: `plot_05_tabnet_results.png`, `plot_05_tabnet_attention.png`

### 6C: Stacking Ensemble
1. Base learners: LogReg, RF, XGBoost, LightGBM, CatBoost (from PART 3)
2. Generate out-of-fold predictions via 5-fold CV
3. Meta-learner: LogReg (classification) or Ridge (regression) trained on stacked OOF predictions
4. Evaluate stacking ensemble on test set:
   - **Classification**: F1 + AUC with bootstrapped 95% CIs
   - **Regression**: RMSE + R2 + MAE with bootstrapped 95% CIs
5. Save: `plot_05_stacking_results.png`

### 6D: Categorical Feature Handling
- LogReg / SVM / MLP: one-hot encoding (low cardinality) or target encoding (high cardinality)
- RF / XGBoost / LightGBM: ordinal encoding (tree-based models handle ordinals naturally)
- CatBoost: native categorical handling via `cat_features` parameter
- TabNet: label encoding (handles categoricals via learned embeddings)

### Quality: Code style variation
Apply per `reference/specs/code-style-variation.md`:
- Pick variable naming pattern (A-E)
- Pick comment style (A-E)
- Pick matplotlib style (A-D)
- Pick color palette (A-E)
- Randomize import order
- Record style vector for consistency

## Validation Checkpoint

- [ ] MLP trained with random search, best params reported
- [ ] TabNet trained with random search, best params reported
- [ ] Stacking ensemble built with 5-fold CV, meta-learner trained
- [ ] Appropriate metrics with 95% CIs for all 3 advanced models
- [ ] Results plots for MLP, TabNet, and stacking
- [ ] TabNet attention-based feature importance extracted
- [ ] Categorical features handled appropriately per model type
- [ ] Code style variation applied

## Data Out -> 07-compare-models.md

```
dl_code_py: [PART 5 Python code]
methods_para_mlp: [prose]
methods_para_tabnet: [prose]
methods_para_stacking: [prose]
results_para_mlp: [prose]
results_para_tabnet: [prose]
results_para_stacking: [prose]
tables: [per-model metrics tables]
plots: [results per DL model + stacking]
style_vector: [e.g., "B-A-C-D-E-2-1"]
```
