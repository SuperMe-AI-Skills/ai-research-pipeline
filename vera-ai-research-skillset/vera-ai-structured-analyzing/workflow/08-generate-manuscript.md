# 08 --- Generate Manuscript

## Executor: Main Agent (assembly)

## Data In: All code, prose, tables, plots, and style_vector from steps 04-07

## Assemble methods.md

### Structure (order fixed, content varies)

```markdown
## Methods

### Data Preprocessing
[Para 1: Missing value handling, imputation strategy, outlier treatment]

### Feature Engineering
[Para 2: Encoding strategy (one-hot, target, ordinal, native categorical), scaling method, feature selection if applied]

### Machine Learning Models
[Para 3: LogReg specification + hyperparameter search strategy]
[Para 4: SVM, RF, XGBoost, LightGBM, CatBoost specification; note CatBoost native categorical handling]

### Deep Learning Models
[Para 5: MLP architecture, hidden layers, activation, dropout, batch norm, training strategy]
[Para 6: TabNet architecture, n_steps, n_d/n_a, sparsity, attention mechanism]

### Stacking Ensemble
[Para 7: Base learners, out-of-fold CV strategy, meta-learner specification]

### Evaluation
[Para 8: Classification metrics (F1, AUC) or regression metrics (RMSE, R2, MAE), bootstrapped CIs, train/val/test protocol]

### Model Comparison
[Para 9: Unified importance normalization, cross-method synthesis approach]

### Software
[Para 10: Python version, key packages with versions]
```

### Rules
- Write as if a human analyst chose these methods for THIS specific study
- Never expose pipeline logic or decision rules
- State what was done + why + key parameters
- No results in methods; no code in methods
- Cite methodological references where appropriate
- Follow `reference/rules/reporting-standards.md`

### Quality: Methods variation
- Each paragraph selects one framing from 3 options
- Vary passive vs active voice across paragraphs
- Vary whether hyperparameter ranges are stated or summarized

## Assemble results.md

### Section ordering logic

Choose ONE ordering based on research question emphasis:
- **Order A (benchmark-driven):** Baseline -> ML models -> DL models -> Stacking -> Comparison
- **Order B (complexity-driven):** Stacking -> TabNet -> MLP -> Gradient Boosting -> Linear -> Comparison
- **Order C (interpretability-driven):** LogReg -> Importance -> RF/XGB -> CatBoost -> DL -> Stacking -> Comparison

### Rules
- All metrics are final computed values
- Apply sentence bank rotation
- Include 1-2 cross-references between sections
- Tables and figures referenced by number
- Follow `reference/rules/reporting-standards.md`
- Classification and regression results use appropriate metric language

## Generate references.bib

- Include ONLY references actually cited
- BibTeX format
- Include: methodological, software, reporting guideline citations

## Apply code style variation

Apply style_vector to final code.py per `reference/specs/code-style-variation.md`.

## Validation Checkpoint

- [ ] methods.md contains no results or numbers
- [ ] methods.md covers all analysis steps
- [ ] results.md section order matches research question type
- [ ] All numbers are computed, no placeholders
- [ ] Table/figure references match actual files
- [ ] Metric formatting follows reporting-standards.md
- [ ] Classification: F1/AUC with 95% CIs throughout
- [ ] Regression: RMSE/R2/MAE with 95% CIs throughout
- [ ] references.bib includes all cited works only
- [ ] Code style variation applied to final code.py
- [ ] No meta-commentary about pipeline structure

## Data Out -> Final Deliverables

```
Deliverables:
├── structured_analysis.py         (PARTS 0-6, style-varied)
├── methods.md                     (manuscript Methods section)
├── results.md                     (manuscript Results section)
├── references.bib                 (cited works only)
├── tables/
│   ├── performance_table.csv
│   ├── importance_table.csv
│   └── comparison_table.csv
└── figures/
    ├── plot_01_target_distribution.png
    ├── plot_02_baseline_results.png
    ├── plot_03_*.png              (ML model plots)
    ├── plot_04_subgroup_*.png     (if applicable)
    ├── plot_05_*.png              (DL + stacking plots)
    └── plot_06_comparison.png
```
