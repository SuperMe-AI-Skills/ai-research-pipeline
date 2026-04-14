# Sentence Bank --- Structured/Tabular Data Results

## Purpose

Provide varied phrasings for each type of ML/DL result. Select contextually
per generation. Never repeat patterns within a single document.

---

## Target Distribution (Classification)

B1: "The target variable [label] contained [n] instances across [k] classes. The minority class ([label]) comprised [pct]% of the sample, indicating [adequate/imbalanced] representation."

B2: "Of the [N] observations, [n] ([pct]%) belonged to [class]. The class distribution was [balanced/skewed], with the least represented class at [pct]%."

B3: "Class distribution analysis revealed [pct]% [class1], [pct]% [class2], and [pct]% [class3]. This [supports standard training / warrants class-weighted loss functions]."

B4: "The dataset comprised [N] records distributed across [k] classes. The smallest class contained [n] samples ([pct]%), [above/below] the threshold for reliable per-class evaluation."

---

## Target Distribution (Regression)

R1: "The target variable [name] had a mean of [val] (SD = [val]), with values ranging from [min] to [max]. The distribution was [approximately normal / right-skewed / left-skewed / bimodal]."

R2: "Descriptive statistics for [target] revealed a median of [val] (IQR: [val]-[val]), with [n] observations. [No / Moderate / Severe] skewness (skew = [val]) was observed."

R3: "The continuous outcome [name] (N = [n]) ranged from [min] to [max] (M = [val], SD = [val]). The [Shapiro-Wilk / Kolmogorov-Smirnov] test [suggested / did not suggest] departure from normality (p [</>] .05)."

R4: "Across [N] observations, [target] had a mean of [val] and standard deviation of [val]. Inspection of the distribution [supported / suggested caution regarding] the use of squared-error loss."

---

## ML Model Performance (Classification)

M1: "The [model] classifier achieved a weighted F1 of [val] (95% CI [[L], [U]]) and macro AUC of [val] (95% CI [[L], [U]]) on the held-out test set."

M2: "[Model] classification yielded F1 = [val], 95% CI [[L], [U]], with macro AUC = [val] (95% CI [[L], [U]]). The [best/worst] per-class performance was observed for [class] (F1 = [val])."

M3: "On the test partition (*n* = [val]), the [model] achieved F1 = [val] (95% CI [[L], [U]]). Discrimination was [excellent/good/fair], AUC = [val] (95% CI [[L], [U]])."

M4: "The best [model] configuration ([params]) produced a weighted F1 of [val] and AUC of [val]. Bootstrapped confidence intervals confirmed [stable/variable] performance."

---

## ML Model Performance (Regression)

MR1: "The [model] regressor achieved RMSE = [val] (95% CI [[L], [U]]) and R2 = [val] (95% CI [[L], [U]]) on the held-out test set."

MR2: "[Model] regression yielded RMSE = [val], 95% CI [[L], [U]], explaining [val]% of variance (R2 = [val], 95% CI [[L], [U]]). MAE was [val] (95% CI [[L], [U]])."

MR3: "On the test partition (*n* = [val]), the [model] achieved RMSE = [val] (95% CI [[L], [U]]) with R2 = [val]. Residual analysis [supported / raised concerns about] model assumptions."

MR4: "The best [model] configuration ([params]) produced an RMSE of [val] and R2 of [val]. Bootstrapped intervals confirmed [stable/variable] predictive accuracy."

---

## Deep Learning Performance

D1: "The [model] architecture achieved [F1 = [val] / RMSE = [val]] (95% CI [[L], [U]]) after [epochs] epochs of training (best epoch: [val]). [Macro AUC was [val] / R2 was [val]] (95% CI [[L], [U]])."

D2: "Training [model] on the tabular dataset yielded [F1 = [val] / RMSE = [val]], 95% CI [[L], [U]], with early stopping at epoch [val]. The model [outperformed/matched/underperformed] the ML baseline."

D3: "[Model] produced [a weighted F1 of [val] / an RMSE of [val]] (95% CI [[L], [U]]). The [architecture-specific insight, e.g., attention-based feature selection / learned feature interactions] contributed to [observation]."

D4: "After hyperparameter search ([n] configurations), the best [model] achieved [F1 = [val] / RMSE = [val]] on the test set. The 95% CI [[L], [U]] [overlaps with / is distinct from] the baseline interval."

---

## Stacking Ensemble

S1: "The stacking ensemble combining [n] base learners with a [meta-learner] meta-learner achieved [F1 = [val] / RMSE = [val]] (95% CI [[L], [U]]), [matching/exceeding] the best individual model."

S2: "Stacking with 5-fold out-of-fold predictions yielded [F1 = [val] / RMSE = [val]] (95% CI [[L], [U]]). The meta-learner weights indicated strongest reliance on [model1] and [model2]."

S3: "The ensemble approach produced [F1 = [val] / RMSE = [val]], with the 95% CI [[L], [U]] [overlapping with / narrower than] the best single model's interval, suggesting [complementary / redundant] base learner contributions."

S4: "Combining predictions from [models] via a [meta-learner] meta-learner yielded [metric = [val]]. The modest [improvement/difference] over the best individual model is consistent with the [high/moderate] correlation among base learner predictions."

---

## Feature Importance

I1: "Across ML methods, [feature1] and [feature2] consistently ranked among the top predictors. The unified importance table (Table [X]) shows [pattern]."

I2: "Feature importance analysis revealed convergent rankings: [feature1] (consensus rank [val]) and [feature2] (consensus rank [val]) dominated across coefficient-based, tree-based, and permutation methods."

I3: "The most predictive features were [feature1] (relative importance: [val]/100) and [feature2] ([val]/100), with consistent rankings across [n] methods."

I4: "[Feature1] emerged as the strongest predictor in [n] of [n] methods examined. The convergence of linear coefficients, Gini importance, and gradient boosting gain strengthens this finding."

I5: "TabNet attention masks identified [feature1] as receiving the highest average attention weight ([val]), corroborating the ML-based importance rankings."

---

## Model Comparison Synthesis

C1: "Across all nine modeling approaches, [key finding]. Traditional ML models provided interpretable feature weights, while deep learning captured [nonlinear interactions / attention-weighted patterns]."

C2: "The convergence of linear, tree-based, and neural methods strengthens confidence in [finding]. TabNet additionally provided per-instance feature selection via attention masks."

C3: "Taken together, [finding] was robust across ML and DL paradigms. Logistic regression is recommended for interpretability, supplemented by [model] for [specific insight]."

C4: "Performance was [similar/varied] across model families ([metric] range: [min]-[max]). The unified importance table confirmed that [feature] was the primary predictor regardless of model architecture."

C5: "The stacking ensemble [marginally improved / did not meaningfully improve] upon the best individual model, suggesting that [complementary patterns / redundant information] drove base learner predictions."
