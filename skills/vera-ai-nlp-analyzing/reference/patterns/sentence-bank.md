# Sentence Bank --- NLP Text Classification Results

## Purpose

Provide varied phrasings for each type of ML/DL result. Select contextually
per generation. Never repeat patterns within a single document.

---

## Class Balance

B1: "The target variable [label] contained [n] instances across [k] classes. The minority class ([label]) comprised [pct]% of the sample, indicating [adequate/imbalanced] representation."

B2: "Of the [N] text samples, [n] ([pct]%) were labeled as [class]. The class distribution was [balanced/skewed], with the least represented class at [pct]%."

B3: "Class distribution analysis revealed [pct]% [class1], [pct]% [class2], and [pct]% [class3]. This [supports standard training / warrants class-weighted loss functions]."

B4: "The dataset comprised [N] documents distributed across [k] classes. The smallest class contained [n] samples ([pct]%), [above/below] the threshold for reliable per-class evaluation."

---

## ML Model Performance

M1: "The [model] classifier achieved a weighted F1 of [val] (95% CI [[L], [U]]) and macro AUC of [val] (95% CI [[L], [U]]) on the held-out test set."

M2: "[Model] classification yielded F1 = [val], 95% CI [[L], [U]], with macro AUC = [val] (95% CI [[L], [U]]). The [best/worst] per-class performance was observed for [class] (F1 = [val])."

M3: "On the test partition (*n* = [val]), the [model] achieved F1 = [val] (95% CI [[L], [U]]). Discrimination was [excellent/good/fair], AUC = [val] (95% CI [[L], [U]])."

M4: "The best [model] configuration ([params]) produced a weighted F1 of [val] and AUC of [val]. Bootstrapped confidence intervals confirmed [stable/variable] performance."

---

## Deep Learning Performance

D1: "The [model] architecture achieved F1 = [val] (95% CI [[L], [U]]) after [epochs] epochs of training (best epoch: [val]). Macro AUC was [val] (95% CI [[L], [U]])."

D2: "Fine-tuning [model] on the training corpus yielded F1 = [val], 95% CI [[L], [U]], with early stopping at epoch [val]. The model [outperformed/matched/underperformed] the ML baseline."

D3: "[Model] classification produced a weighted F1 of [val] (95% CI [[L], [U]]) and AUC of [val]. The [architecture-specific insight, e.g., bidirectional processing / multi-scale filters / contextual embeddings] contributed to [observation]."

D4: "After hyperparameter search ([n] configurations), the best [model] achieved F1 = [val] on the test set. The 95% CI [[L], [U]] [overlaps with / is distinct from] the baseline interval."

---

## Feature Importance

I1: "Across ML methods, [feature1] and [feature2] consistently ranked among the top predictors. The unified importance table (Table [X]) shows [pattern]."

I2: "Feature importance analysis revealed convergent rankings: [feature1] (consensus rank [val]) and [feature2] (consensus rank [val]) dominated across logistic, tree-based, and permutation methods."

I3: "The most discriminative features were [feature1] (relative importance: [val]/100) and [feature2] ([val]/100), with consistent rankings across [n] methods."

I4: "[Feature1] emerged as the strongest predictor in [n] of [n] methods examined. The convergence of linear coefficients and tree-based importance strengthens this finding."

---

## Model Comparison Synthesis

C1: "Across all seven classification approaches, [key finding]. Traditional ML models provided interpretable feature weights, while deep learning captured [contextual/sequential] patterns."

C2: "The convergence of linear, ensemble, and neural methods strengthens confidence in [finding]. ALBERT additionally leveraged pre-trained language representations for [observation]."

C3: "Taken together, [finding] was robust across ML and DL paradigms. The logistic regression is recommended for interpretability, supplemented by [model] for [specific insight]."

C4: "Performance was [similar/varied] across model families (F1 range: [min]-[max]). The unified importance table confirmed that [feature] was the primary discriminator regardless of model architecture."
