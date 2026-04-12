# Output Quality Variation Protocol

Read this file before every generation. Apply all variation layers for
natural, diverse, non-repetitive output.

## Layer 1: Phrasing Variation (Sentence Bank)

For each CNN/ViT result, maintain 4-6 alternative phrasings. Select
contextually per generation. Never repeat the same phrasing pattern for
the same type of result within a single document.

See `reference/patterns/sentence-bank.md` for the full bank.

Rules:
- Rotate across paragraphs within the same document
- Select based on data context (model type, metric values, class names)
- Someone reading 10 outputs should see 10 different interpretive choices

## Layer 2: Structure Variation

### Section ordering (within scientific validity constraints):

Choose ONE ordering for results.md based on research question:
- **Order A (benchmark-driven):** Baseline -> CNN models -> ViT -> Ensemble -> Comparison
- **Order B (architecture-driven):** Transformer -> ResNet -> EfficientNet -> DenseNet -> Ensemble -> Comparison
- **Order C (interpretability-driven):** GradCAM -> CNN performance -> Attention maps -> Ensemble -> Comparison

### Table and figure naming:
Vary: "Table 1. Model Performance Comparison" vs
"Table 1. Classification Results Across Architectures" vs
"Table 1. Image Classification Benchmarks"

### Figure layout:
Side-by-side vs stacked, grid vs individual --- vary across generations.
GradCAM overlays: 2x4 grid vs single-column vs grouped by class.

## Layer 3: Interpretation Depth Variation

Randomly include 1-2 of the following per analysis section:
- Practical significance framing ("the F1 improvement of 0.03 corresponds to...")
- Comparison to published benchmarks ("consistent with ImageNet transfer learning results on similar tasks")
- Limitation acknowledgment inline ("though the small validation set limits...")
- Methodological justification ("DenseNet121 was chosen for its suitability in medical imaging")
- Efficiency framing ("EfficientNet-B0 achieved comparable performance with 5x fewer parameters")
- Interpretability framing ("GradCAM highlighted regions consistent with domain expert annotations")

Generate contextually from actual data. These are NOT templates.

## Layer 4: Code Style Variation

See `reference/specs/code-style-variation.md` for the 7-dimension specification.

Apply per-generation variations to:
- Variable naming patterns
- Comment styles
- Section separators
- matplotlib/seaborn styles
- Color palettes
- Import order
- Function organization

## Layer 5: System Capabilities

What this system automates end-to-end:
- Correct architecture selection adapting to image characteristics
- Class balance handling that triggers appropriate strategies
- Transfer learning strategy (feature extraction vs full fine-tuning) based on dataset size
- Cross-method comparison with GradCAM and attention map synthesis
- Feature attribution unification across CNN and Transformer architectures
- Consistent reporting across CNN and ViT paradigms
- Data augmentation pipeline tailored to image domain
- Tabular metadata integration (image + metadata fusion)
- Medical imaging support (DICOM, DenseNet rationale)
- Citation accuracy and completeness
