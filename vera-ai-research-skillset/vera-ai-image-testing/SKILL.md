---
name: vera-ai-image-testing
description: >-
  Runs data quality diagnostics and baseline image classification. Produces
  class distribution tables, sample image grids, image size and channel
  statistics, a baseline CNN or pre-trained feature extractor with Logistic
  Regression, weighted F1 and macro AUC (bootstrapped 95% CIs), confusion
  matrix, and ROC curves. Ends with a recommendation block listing additional
  models available in the analysis workflow. Outputs Python scripts with 2
  publication-quality plots. Triggered when user has image data and says
  "image classification," "computer vision," "CNN," "object recognition,"
  "medical imaging," "image detection," "visual recognition," "photo
  classification," "X-ray classification," or describes a task involving
  classifying images into categories. Does not handle free-text NLP or
  tabular-only data.
user-invocable: true
allowed-tools: Read, Bash, Write, Edit
---

# Image Classification --- Data Diagnostics & Baseline Modeling

Open-source skill.

## Workflow

Read each step file in `workflow/` before executing that step.

| Step | File | Executor | Output |
|---|---|---|---|
| Collect | `workflow/01-collect-inputs.md` | Main Agent | Structured input summary |
| Diagnose | `workflow/02-check-distribution.md` | Main Agent | PART 1 code block |
| Baseline | `workflow/03-run-primary-test.md` | Main Agent | PART 2-3 code blocks + T1 track artifacts |

## Decision Tree

```
1. CHECK CLASS BALANCE
   ├── Balanced (minority ≥ 10%) → standard methods
   └── Imbalanced (minority < 10%) → weighted loss, data augmentation note

2. IMAGE SIZE STRATEGY
   ├── Uniform size → use directly
   └── Variable size → resize to common dimension (224x224 default)

3. BASELINE MODEL SELECTION
   ├── N ≥ 1000 → Simple CNN (train from scratch)
   └── N < 1000 → Pre-trained feature extractor (ResNet18) + LogReg
```

## Required Inputs

| Role | What to collect |
|---|---|
| **Image source** | Directory path, dataset name, or file format |
| **Label source** | Subdirectory names, CSV mapping, or metadata |
| **Image format** | PNG, JPEG, DICOM, etc. |
| **Number of classes** | Binary or multi-class |

## Code Structure

```
PART 0: Setup & Data Loading
PART 1: Data Diagnostics          → plot_01_data_overview.png
PART 2: Baseline Classification   → plot_02_confusion_roc.png
PART 3: Recommendation Block      → text pointing to analysis workflow
```

## Reporting Standards

1. Metrics: weighted F1 and macro AUC (OVR) — always with 95% bootstrapped CIs
2. Format: "F1 = 0.XXX, 95% CI [0.XXX, 0.XXX]"
3. AUC: "AUC = 0.XXX, 95% CI [0.XXX, 0.XXX]"
4. Decimal places: 3 for F1/AUC, 1 for percentages, 0 for counts
5. Image dimensions: report as HxWxC
6. Sample size: report final analytic N (train/val/test split sizes)
7. Data augmentation: always report what augmentations were applied
8. Bootstrapped CIs: 1000 iterations, 2.5th/97.5th percentiles

## Models Available

| Workflow | Models |
|---|---|
| Testing (this skill) | Simple CNN or ResNet18 feature extractor + LogReg |
| Analysis (vera-ai-image-analyzing) | ResNet50, EfficientNet, VGG16, ViT, DenseNet, Ensemble |

## Example Dataset

torchvision built-ins: `CIFAR10`, `FashionMNIST`, `MNIST`.
Python: `from torchvision.datasets import CIFAR10`

## Cross-Skill Interface

```
Output:
├── code_python      → .py script
├── methods_md       → methods.md baseline fragment
├── results_md       → results.md baseline results fragment
├── tables/          → Markdown/CSV tables for dataset stats + metrics
├── figures/         → 2 PNGs (data overview + confusion/ROC)
├── references_bib   → .bib with baseline/evaluation citations
└── recommendations  → text block (what analysis workflow produces)
```

When this skill is used as `T1_baseline` inside `vera-ai-application-pipeline`,
the standardized track artifacts above are REQUIRED, not optional.
