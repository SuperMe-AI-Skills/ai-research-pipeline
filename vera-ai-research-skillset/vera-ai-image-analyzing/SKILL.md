---
name: vera-ai-image-analyzing
description: >-
  Server-side extension that completes the full analysis pipeline for image
  classification after vera-ai-image-testing has run. Adds ResNet50,
  EfficientNet-B0, VGG16, DenseNet121 CNN classifiers with transfer learning
  (feature extraction and full fine-tuning), subgroup analysis by metadata
  or image properties, advanced models (ViT transformer with optional
  tabular fusion and hyperparameter search), ensemble methods,
  cross-method comparison with unified feature attribution (GradCAM,
  attention maps) on a 0-100 scale, and manuscript-ready methods.md and
  results.md. Applies output variation and code style diversity for
  natural, non-repetitive output. Open-source skill. Triggered after
  vera-ai-image-testing completes, or direct API call with an image
  classification task.
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob
---

# Image Classification --- Full Analysis & Manuscript Generation

Open-source skill. Read `reference/specs/output-variation-protocol.md`
before every generation --- apply all variation layers for natural, diverse output.

## Workflow

Continues from where vera-ai-image-testing stopped (PART 0-2 done).

| Step | File | Executor | Output |
|---|---|---|---|
| Additional CNN models | `workflow/04-run-additional-models.md` | Main Agent | PART 3 code + prose |
| Subgroup | `workflow/05-analyze-subgroups.md` | Main Agent | PART 4 code + prose |
| Advanced models | `workflow/06-fit-advanced-models.md` | Main Agent | PART 5 code + prose |
| Comparison | `workflow/07-compare-models.md` | Main Agent | PART 6 code + prose |
| Manuscript | `workflow/08-generate-manuscript.md` | Main Agent | methods.md + results.md |

## Additional Inputs

Collect if not already provided:
- Target discipline (for reporting conventions)
- Target journal or style (CVPR, ECCV, MICCAI, NeurIPS, etc.)
- Research question / hypothesis
- Subgroup variable or image property for stratification
- Image dimensions (HxWxC) and any DICOM metadata if medical imaging

## Output Structure

```
output/
├── methods.md
├── results.md
├── tables/             <- Markdown + CSV per table
├── figures/            <- PNGs, 300 DPI
├── references.bib
└── code.py             <- Style-varied
```

## Key References (read before generation)

| File | Purpose |
|---|---|
| `reference/specs/output-variation-protocol.md` | Output quality variation layers |
| `reference/specs/code-style-variation.md` | Seven-dimension code style diversity |
| `reference/patterns/sentence-bank.md` | 4-6 phrasings per result type |
| `reference/rules/reporting-standards.md` | Hard rules for CNN/ViT reporting |

## Reporting Standards

Same as vera-ai-image-testing, plus:
- All models: report F1 (weighted) and AUC (macro) with bootstrapped 95% CIs
- Transfer learning: report backbone name, pre-trained weights, frozen vs fine-tuned layers
- Training: report epochs, best epoch, learning rate, batch size, data augmentation pipeline
- ViT: report pre-trained model name, patch size, whether backbone was frozen
- Image input: report HxWxC dimensions, normalization (mean/std), parameter counts, FLOPs
- Feature attribution: unified 0-100 scale via GradCAM (CNNs) and attention maps (ViT)
- Model comparison: frame as convergent findings, not horse race
- Small dataset CNN: frame as "exploratory"; never claim generalizability
- Medical imaging: report DICOM preprocessing, windowing, DenseNet rationale

## Cross-Skill Interface

```
Method Unit Contract:
├── code_python      -> .py script (style-varied)
├── methods_md       -> methods.md (varied structure)
├── results_md       -> results.md (varied phrasing)
├── tables/          -> Markdown + CSV
├── figures/         -> PNGs 300 DPI (varied layout)
├── references_bib   -> .bib with cited references
└── comparison       -> cross-method narrative (in results.md)
```
