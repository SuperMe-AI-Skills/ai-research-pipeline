# Code Style Variation Specification

## Purpose

Every code generation produces naturally varied surface patterns ---
different variable names, comments, colors, and import orders --- so
outputs look hand-written rather than templated, while preserving
identical analytical logic.

## Style Dimensions

### 1. Variable Naming Pattern (pick ONE per generation)

| Pattern | ResNet50 | EfficientNet | VGG16 | DenseNet121 | ViT | Ensemble |
|---|---|---|---|---|---|---|
| A | `model_resnet` | `model_effnet` | `model_vgg` | `model_densenet` | `model_vit` | `model_ensemble` |
| B | `clf_resnet` | `clf_effnet` | `clf_vgg` | `clf_densenet` | `net_vit` | `clf_ensemble` |
| C | `resnet_fit` | `effnet_fit` | `vgg_fit` | `densenet_fit` | `vit_fit` | `ensemble_fit` |
| D | `rn50_model` | `eb0_model` | `vgg_model` | `dn121_model` | `vit_model` | `ens_model` |
| E | `img_resnet` | `img_effnet` | `img_vgg` | `img_densenet` | `img_vit` | `img_ensemble` |

### 2. Comment Style (pick ONE per generation)

```
Style A: # -- Section Name ------------------------------------------
Style B: # --- Section Name ---
Style C: # == Section Name ==
Style D: # Section Name
Style E: # > Section Name
```

### 3. Section Separator (pick ONE per generation)

```
Sep A: print("=" * 60)
Sep B: print("-" * 60)
Sep C: print("#" * 60)
Sep D: print("\n---\n")
```

### 4. Matplotlib Style (pick ONE, maintain within script)

| Style | Python |
|---|---|
| A | `plt.style.use('seaborn-v0_8-whitegrid')` |
| B | `plt.style.use('seaborn-v0_8-white')` |
| C | `plt.style.use('classic')` with grid off |
| D | `plt.style.use('seaborn-v0_8-ticks')` |

### 5. Color Palette (pick ONE per generation)

| Palette | Primary | Secondary | Accent | Group3 |
|---|---|---|---|---|
| A | `#4A90D9` | `#D94A4A` | `#D9A04A` | `#4AD97A` |
| B | `#2C7BB6` | `#D7191C` | `#FDAE61` | `#ABD9E9` |
| C | `#1B9E77` | `#D95F02` | `#7570B3` | `#E7298A` |
| D | `#E41A1C` | `#377EB8` | `#4DAF4A` | `#984EA3` |
| E | `#66C2A5` | `#FC8D62` | `#8DA0CB` | `#E78AC3` |

### 6. Import Order Randomization

```python
# Group 1 (data): pandas, numpy --- either order
# Group 2 (vision): torchvision, PIL, cv2 --- any order
# Group 3 (models): torch, torchvision.models, timm --- torch first always
# Group 4 (viz): matplotlib, seaborn --- either order
# Group 5 (eval): sklearn.metrics, scipy --- either order
# Group 6 (interp): pytorch_grad_cam, captum --- either order
```

### 7. Function Organization

Vary order of helper function definitions (those without dependencies):
- Evaluation helpers before or after training functions
- Data augmentation setup at top or inline
- GradCAM utility functions early or late

## Application

At code generation time:
1. Pick one option from each dimension (7 choices)
2. Apply consistently throughout the entire script

## What Style Variation Does NOT Change

- Analytical logic (same models, same hyperparameters, same evaluation)
- Numerical output (same metrics, same CIs)
- Package/function calls (same API, same arguments)
- File naming convention (plot_01, plot_02, etc.)
