# 04 --- Run Additional CNN Models

## Executor: Main Agent (code generation)

## Data In: PART 0-2 code from vera-ai-image-testing output

## Prerequisite: Testing workflow (PART 0-2) already executed.

## Generate code for PART 3: Additional CNN Models

### For each CNN model (ResNet50, EfficientNet-B0, VGG16, DenseNet121):

1. **Transfer learning search** over model-specific hyperparameters (see config/default.json)
   - Phase 1: Feature extraction (frozen backbone, train head only)
   - Phase 2: Full fine-tuning (unfreeze backbone with lower LR)
2. **Data augmentation**: apply training augmentation pipeline (RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop, RandomAffine)
3. **Select best** by validation weighted F1
4. **Report best hyperparameters** including frozen/unfrozen status, learning rates
5. **Report model specs**: parameter count (total, trainable), FLOPs, input dimensions (HxWxC)
6. **Evaluate on test set**:
   - Classification report (precision, recall, F1 per class)
   - Bootstrapped F1 (1000 iterations, 95% CI)
   - Bootstrapped AUC (macro OVR, 1000 iterations, 95% CI)
   - Confusion matrix
   - Multi-class ROC curves
7. **Feature attribution** (GradCAM for all CNNs):
   - ResNet50: GradCAM on layer4[-1]
   - EfficientNet-B0: GradCAM on features[-1]
   - VGG16: GradCAM on features[-1]
   - DenseNet121: GradCAM on features.denseblock4
   - Display GradCAM overlays for representative samples (4 per class)
   - All attribution scores normalized 0-100 scale
8. **Save plots**: `plot_03_resnet_confusion_roc.png`, `plot_03_effnet_confusion_roc.png`, `plot_03_vgg_confusion_roc.png`, `plot_03_densenet_confusion_roc.png`, `plot_03_gradcam_samples.png`
9. **Medical imaging** (if DenseNet121 with DICOM): report windowing parameters, preprocessing steps

### Quality: Apply sentence bank from `reference/patterns/sentence-bank.md`
- Vary whether F1 or AUC leads the interpretation sentence
- Vary per-class highlight selection (best class vs worst class first)
- Include 0-1 methodological justifications per model
- Vary whether parameter count or GradCAM finding is highlighted

## Validation Checkpoint

- [ ] All 4 CNN models trained with transfer learning (feature extraction + fine-tuning)
- [ ] Best hyperparameters reported for each (including freeze status, LR schedule)
- [ ] Parameter counts and FLOPs reported for each model
- [ ] F1 + AUC with 95% CIs for each model
- [ ] Classification reports for each model
- [ ] Confusion matrices and ROC curves plotted
- [ ] GradCAM attribution computed for all CNN models
- [ ] Data augmentation pipeline documented
- [ ] Sentence bank applied (no repeated phrasing patterns)

## Data Out -> 05-analyze-subgroups.md

```
additional_models_code_py: [PART 3 Python code]
methods_para_cnn: [methods paragraph prose per model]
results_para_cnn: [results paragraph prose per model]
plots: [list of new plot filenames]
tables: [list of new table data]
gradcam_tables: {resnet: df, efficientnet: df, vgg: df, densenet: df}
```
