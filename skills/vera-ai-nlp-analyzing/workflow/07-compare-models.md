# 07 --- Compare Models

## Executor: Main Agent (code generation)

## Data In: All model outputs from steps 03-06

## Design Principle

Models are analytic lenses, not contestants. Each model type captures different
patterns in text data. The comparison step synthesizes what converges and what
each uniquely reveals. Never frame as "which model is best." No F1 horse race.

## Generate code for PART 6: Cross-Method Insight Synthesis

### 7A: Unified Performance Table

| Model | F1 (weighted) | 95% CI | AUC (macro) | 95% CI | Notes |
|-------|---------------|--------|-------------|--------|-------|
| LogReg (baseline) | ... | ... | ... | ... | TF-IDF |
| SVM | ... | ... | ... | ... | RBF kernel |
| Random Forest | ... | ... | ... | ... | Gini importance |
| LightGBM | ... | ... | ... | ... | Gradient boosting |
| GRU | ... | ... | ... | ... | Bidirectional |
| TextCNN | ... | ... | ... | ... | Multi-filter |
| ALBERT | ... | ... | ... | ... | Fine-tuned transformer |

### 7B: Unified Feature Importance Table (ML models only)

Normalize all importance measures to a 0-100 scale (max = 100):
- LogReg: |coefficients| rescaled
- RF: Gini importance rescaled
- LightGBM: gain importance rescaled
- Permutation importance (SVM): rescaled

| Feature | LogReg | SVM (perm) | RF (Gini) | LightGBM (gain) | Rank Consensus |
|---------|--------|------------|-----------|-----------------|----------------|

Rank consensus = average rank across all methods.

### 7C: Insight Synthesis Table

| Method Family | Unique Insight |
|---------------|----------------|
| Linear (LogReg) | [which features are most discriminative, coefficient directions] |
| Kernel (SVM) | [nonlinear decision boundaries, hard-to-classify regions] |
| Ensemble (RF, LightGBM) | [feature interactions, importance convergence] |
| Sequential (GRU) | [word order patterns, sequential dependencies] |
| N-gram (TextCNN) | [local phrase patterns, filter-size-specific features] |
| Contextual (ALBERT) | [semantic understanding, transfer learning effectiveness] |

### 7D: Narrative Synthesis

3-4 sentences covering:
1. What converges across methods (strongest predictive features agreement)
2. What traditional ML uniquely reveals (interpretable features)
3. What deep learning uniquely reveals (contextual patterns, sequential info)
4. Overall: convergence strengthens confidence in key finding

### Quality: Synthesis variation
Apply sentence bank (model comparison section). Rotate lead-in.

## Validation Checkpoint

- [ ] Unified performance table with all 7 models
- [ ] Unified importance table with ML models on 0-100 scale
- [ ] Rank consensus computed
- [ ] Insight synthesis table: one row per model family
- [ ] No F1 horse-race framing
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
