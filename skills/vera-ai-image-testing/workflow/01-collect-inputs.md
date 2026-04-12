# 01 --- Collect Inputs

## Executor: Main Agent

## Data In: User request (natural language)

## Required

Ask for anything missing. Do not proceed until all required inputs are collected.

1. **Image data** --- one of:
   - Directory with subdirectories per class (ImageFolder format)
   - Directory + CSV/JSON mapping filenames to labels
   - Named dataset (CIFAR10, MNIST, etc.)
   - Uploaded archive (zip/tar) with images

2. **Label source** --- how classes are defined
   - Subdirectory names (most common)
   - CSV column mapping filename → label
   - Confirm number of classes and label meanings

3. **Image format** --- file type and properties
   - Format: PNG, JPEG, TIFF, DICOM, etc.
   - Color: RGB, grayscale, or other
   - Size: uniform or variable dimensions

## Optional (collect for recommendation quality)

4. **Target image size** --- resize dimension (default 224x224)
5. **Domain context** --- medical, satellite, general, etc.
6. **Data augmentation preferences** --- specific augmentations to apply or avoid
7. **Hardware constraints** --- GPU availability, memory limits
8. **Sample size** --- if not evident from directory count

## Validation Checkpoint

- [ ] Image source identified and accessible
- [ ] Labels confirmed with class count and meanings
- [ ] Image format and color channels confirmed
- [ ] If N < 100 per class, small-sample warning issued
- [ ] If images are variable size, resize strategy confirmed
- [ ] At least image source + labels collected

## Data Out -> 02-check-distribution.md

Structured input summary containing:
```
image_source: {path | dataset_name | description}
label_source: {subdirectory | csv_path | dataset}
n_classes: int
class_labels: [label_names]
image_format: {format, channels, size_uniform}
target_size: [H, W]
domain: {medical | satellite | general | ...}
sample_size: N or "unknown"
```
