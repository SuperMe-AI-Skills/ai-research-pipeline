# Sentence Bank --- Image Classification Results

## Purpose

Provide varied phrasings for each type of CNN/ViT result. Select contextually
per generation. Never repeat patterns within a single document.

---

## Class Balance

B1: "The target variable [label] contained [n] instances across [k] classes. The minority class ([label]) comprised [pct]% of the sample, indicating [adequate/imbalanced] representation."

B2: "Of the [N] images, [n] ([pct]%) were labeled as [class]. The class distribution was [balanced/skewed], with the least represented class at [pct]%."

B3: "Class distribution analysis revealed [pct]% [class1], [pct]% [class2], and [pct]% [class3]. This [supports standard training / warrants class-weighted loss functions / motivates oversampling]."

B4: "The dataset comprised [N] images ([HxWxC], [colorspace]) distributed across [k] classes. The smallest class contained [n] samples ([pct]%), [above/below] the threshold for reliable per-class evaluation."

---

## CNN Model Performance

M1: "The [model] classifier achieved a weighted F1 of [val] (95% CI [[L], [U]]) and macro AUC of [val] (95% CI [[L], [U]]) on the held-out test set using [frozen/fine-tuned] ImageNet weights."

M2: "[Model] classification with transfer learning yielded F1 = [val], 95% CI [[L], [U]], with macro AUC = [val] (95% CI [[L], [U]]). The [best/worst] per-class performance was observed for [class] (F1 = [val])."

M3: "On the test partition (*n* = [val]), the [model] ([params]M parameters, [flops]G FLOPs) achieved F1 = [val] (95% CI [[L], [U]]). Discrimination was [excellent/good/fair], AUC = [val] (95% CI [[L], [U]])."

M4: "The best [model] configuration ([frozen/fine-tuned], lr=[val]) produced a weighted F1 of [val] and AUC of [val]. Bootstrapped confidence intervals confirmed [stable/variable] performance."

---

## ViT / Transformer Performance

D1: "The Vision Transformer (ViT-B/16) achieved F1 = [val] (95% CI [[L], [U]]) after [epochs] epochs of fine-tuning (best epoch: [val]). Macro AUC was [val] (95% CI [[L], [U]])."

D2: "Fine-tuning ViT-B/16 on the training images yielded F1 = [val], 95% CI [[L], [U]], with early stopping at epoch [val]. The model [outperformed/matched/underperformed] the CNN baseline."

D3: "The Vision Transformer produced a weighted F1 of [val] (95% CI [[L], [U]]) and AUC of [val]. Global self-attention over [patch_size]x[patch_size] patches captured [long-range spatial dependencies / fine-grained class distinctions]."

D4: "After hyperparameter search ([n] configurations), the best ViT achieved F1 = [val] on the test set. The 95% CI [[L], [U]] [overlaps with / is distinct from] the CNN baseline interval."

---

## Feature Attribution (GradCAM / Attention)

I1: "Across CNN architectures, GradCAM consistently highlighted [region/feature] as the most discriminative area. The unified attribution table (Table [X]) shows [pattern]."

I2: "Feature attribution analysis revealed convergent spatial focus: [region1] (consensus score [val]/100) and [region2] (consensus score [val]/100) dominated across GradCAM and attention-based methods."

I3: "The most discriminative image regions were [region1] (relative attribution: [val]/100) and [region2] ([val]/100), with consistent spatial focus across [n] architectures."

I4: "[Region/feature] emerged as the primary discriminative area in [n] of [n] models examined. The convergence of CNN GradCAM and ViT attention maps strengthens this finding."

---

## Model Comparison Synthesis

C1: "Across all six classification approaches, [key finding]. CNN architectures captured local texture patterns, while the Vision Transformer leveraged global spatial context."

C2: "The convergence of residual, dense, and transformer architectures strengthens confidence in [finding]. ViT additionally captured long-range dependencies via self-attention for [observation]."

C3: "Taken together, [finding] was robust across CNN and Transformer paradigms. EfficientNet-B0 is recommended for deployment efficiency, supplemented by [model] for [specific insight]."

C4: "Performance was [similar/varied] across architecture families (F1 range: [min]-[max]). The unified attribution table confirmed that [region/pattern] was the primary discriminator regardless of model architecture."
