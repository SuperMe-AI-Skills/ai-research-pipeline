# 08 --- Generate Manuscript

## Executor: Main Agent (assembly)

## Data In: All code, prose, tables, plots, and style_vector from steps 04-07

## Assemble methods.md

### Structure (order fixed, content varies)

```markdown
## Methods

### Image Preprocessing
[Para 1: Image resizing, normalization (ImageNet mean/std), channel format (HxWxC)]

### Data Augmentation
[Para 2: Training augmentation pipeline (RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop, RandomAffine), validation/test transforms (Resize + CenterCrop)]

### CNN Models (Transfer Learning)
[Para 3: Baseline CNN specification + transfer learning strategy (feature extraction vs fine-tuning)]
[Para 4: ResNet50, EfficientNet-B0, VGG16, DenseNet121 specification with pre-trained weights, parameter counts]

### Vision Transformer
[Para 5: ViT architecture, patch size, pre-trained weights, fine-tuning strategy]

### Ensemble Methods
[Para 6: Soft voting and stacking ensemble specification, base model selection]

### Interpretability
[Para 7: GradCAM methodology for CNNs (target layers per architecture), attention map extraction for ViT]

### Evaluation
[Para 8: Metrics (F1, AUC), bootstrapped CIs, train/val/test protocol, data augmentation only on train]

### Model Comparison
[Para 9: Unified attribution normalization, cross-method synthesis approach, parameter/FLOPs comparison]

### Software
[Para 10: Python version, PyTorch, torchvision, timm, key packages with versions]
```

### Rules
- Write as if a human analyst chose these methods for THIS specific study
- Never expose pipeline logic or decision rules
- State what was done + why + key parameters
- No results in methods; no code in methods
- Report image dimensions (HxWxC), augmentation details, parameter counts
- Cite methodological references where appropriate
- Follow `reference/rules/reporting-standards.md`

### Quality: Methods variation
- Each paragraph selects one framing from 3 options
- Vary passive vs active voice across paragraphs
- Vary whether hyperparameter ranges are stated or summarized

## Assemble results.md

### Section ordering logic

Choose ONE ordering based on research question emphasis:
- **Order A (benchmark-driven):** Baseline -> CNN models -> ViT -> Ensemble -> Comparison
- **Order B (architecture-driven):** ViT -> ResNet -> EfficientNet -> DenseNet -> VGG -> Ensemble -> Comparison
- **Order C (interpretability-driven):** GradCAM findings -> CNN performance -> ViT attention -> Ensemble -> Comparison

### Rules
- All metrics are final computed values
- Apply sentence bank rotation
- Include 1-2 cross-references between sections
- Tables and figures referenced by number
- Include GradCAM visualizations and attention maps in results
- Follow `reference/rules/reporting-standards.md`

## Generate references.bib

- Include ONLY references actually cited
- BibTeX format
- Include: methodological (ResNet, EfficientNet, ViT papers), software, reporting guideline citations

## Apply code style variation

Apply style_vector to final code.py per `reference/specs/code-style-variation.md`.

## Validation Checkpoint

- [ ] methods.md contains no results or numbers
- [ ] methods.md covers all analysis steps including augmentation and interpretability
- [ ] results.md section order matches research question type
- [ ] All numbers are computed, no placeholders
- [ ] Table/figure references match actual files
- [ ] Metric formatting follows reporting-standards.md
- [ ] F1/AUC with 95% CIs throughout
- [ ] Image dimensions, parameter counts, and FLOPs reported
- [ ] GradCAM and attention map results included
- [ ] references.bib includes all cited works only
- [ ] Code style variation applied to final code.py
- [ ] No meta-commentary about pipeline structure

## Data Out -> Final Deliverables

```
Deliverables:
├── image_analysis.py             (PARTS 0-6, style-varied)
├── methods.md                    (manuscript Methods section)
├── results.md                    (manuscript Results section)
├── references.bib                (cited works only)
├── tables/
│   ├── performance_table.csv
│   ├── attribution_table.csv
│   └── comparison_table.csv
└── figures/
    ├── plot_01_data_overview.png
    ├── plot_02_confusion_roc.png
    ├── plot_03_*.png             (CNN model plots)
    ├── plot_03_gradcam_samples.png
    ├── plot_04_subgroup_*.png    (if applicable)
    ├── plot_05_*.png             (ViT + Ensemble plots)
    ├── plot_05_vit_attention_maps.png
    └── plot_06_comparison.png
```
