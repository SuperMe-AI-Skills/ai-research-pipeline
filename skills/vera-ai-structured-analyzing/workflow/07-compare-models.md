# 07 --- Compare Models

## Executor: Main Agent (code generation)

## Data In: All model outputs from steps 03-06

## Design Principle

Models are analytic lenses, not contestants. Each model type captures different
patterns in structured data. The comparison step synthesizes what converges and
what each uniquely reveals. Never frame as "which model is best." No metric
horse race.

## Generate code for PART 6: Cross-Method Insight Synthesis

### 7A: Unified Performance Table

#### Classification:

| Model | F1 (weighted) | 95% CI | AUC (macro) | 95% CI | Notes |
|-------|---------------|--------|-------------|--------|-------|
| LogReg (baseline) | ... | ... | ... | ... | L1/L2 |
| SVM | ... | ... | ... | ... | RBF kernel |
| Random Forest | ... | ... | ... | ... | Gini importance |
| XGBoost | ... | ... | ... | ... | Gradient boosting |
| LightGBM | ... | ... | ... | ... | Leaf-wise growth |
| CatBoost | ... | ... | ... | ... | Native categoricals |
| MLP | ... | ... | ... | ... | Neural network |
| TabNet | ... | ... | ... | ... | Attention-based |
| Stacking Ensemble | ... | ... | ... | ... | Meta-learner |

#### Regression:

| Model | RMSE | 95% CI | R2 | 95% CI | MAE | 95% CI | Notes |
|-------|------|--------|-----|--------|-----|--------|-------|
| LinReg (baseline) | ... | ... | ... | ... | ... | ... | OLS/Ridge |
| SVM (SVR) | ... | ... | ... | ... | ... | ... | RBF kernel |
| Random Forest | ... | ... | ... | ... | ... | ... | Gini importance |
| XGBoost | ... | ... | ... | ... | ... | ... | Gradient boosting |
| LightGBM | ... | ... | ... | ... | ... | ... | Leaf-wise growth |
| CatBoost | ... | ... | ... | ... | ... | ... | Native categoricals |
| MLP | ... | ... | ... | ... | ... | ... | Neural network |
| TabNet | ... | ... | ... | ... | ... | ... | Attention-based |
| Stacking Ensemble | ... | ... | ... | ... | ... | ... | Meta-learner |

### 7B: Unified Feature Importance Table (ML models only)

Normalize all importance measures to a 0-100 scale (max = 100):
- LogReg: |coefficients| rescaled
- SVM: permutation importance rescaled
- RF: Gini importance rescaled
- XGBoost: gain importance rescaled
- LightGBM: gain importance rescaled
- CatBoost: gain importance rescaled

| Feature | LogReg | SVM (perm) | RF (Gini) | XGBoost (gain) | LightGBM (gain) | CatBoost (gain) | Rank Consensus |
|---------|--------|------------|-----------|----------------|-----------------|-----------------|----------------|

Rank consensus = average rank across all methods.

### 7C: Insight Synthesis Table

| Method Family | Unique Insight |
|---------------|----------------|
| Linear (LogReg) | [which features are most discriminative, coefficient directions, monotonic effects] |
| Kernel (SVM) | [nonlinear decision boundaries, hard-to-classify/predict regions] |
| Tree Ensemble (RF) | [feature interactions via Gini, robust to outliers] |
| Gradient Boosting (XGB/LGB/CatBoost) | [feature interactions, gain convergence, categorical handling] |
| Neural Network (MLP) | [nonlinear feature combinations, learned representations] |
| Attention-based (TabNet) | [per-instance feature selection, attention mask patterns] |
| Stacking Ensemble | [complementary model strengths, meta-learner weights] |

### 7D: Narrative Synthesis

3-4 sentences covering:
1. What converges across methods (strongest predictive features agreement)
2. What traditional ML uniquely reveals (interpretable features, coefficient directions)
3. What deep learning uniquely reveals (nonlinear interactions, attention patterns)
4. Overall: convergence strengthens confidence in key finding

### Quality: Synthesis variation
Apply sentence bank (model comparison section). Rotate lead-in.

## Validation Checkpoint

- [ ] Unified performance table with all 9 models (classification) or 9 models (regression)
- [ ] Unified importance table with ML models on 0-100 scale
- [ ] Rank consensus computed
- [ ] Insight synthesis table: one row per model family
- [ ] No metric horse-race framing
- [ ] Narrative synthesis: 3-4 sentences
- [ ] Sentence bank applied

## Data Out -> 08-generate-manuscript.md

```
comparison_code_py: [PART 6 Python code]
unified_performance_table: [all models]
unified_importance_table: [ML models, 0-100]
insight_table: [model family x unique insight]
results_para_comparison: [synthesis paragraph prose]
```
