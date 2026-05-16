# Skill Routing Table

Maps confirmed data modality to the paired reviewing and generating skills.

All paths are relative to `REPO_ROOT`, the directory that contains the extracted
`vera-ai-research-skillset/` contents. In this repository and in a direct Codex
install, the skills are flat sibling directories:

```text
vera-ai-nlp-reviewing/
vera-ai-nlp-generating/
vera-ai-structured-reviewing/
vera-ai-structured-generating/
vera-ai-image-reviewing/
vera-ai-image-generating/
```

Do not prepend `vera-ai-analysis-engine/`; that was the older source-tree layout.

## Routing Table

| Modality | Reviewing Skill Directory | Generating Skill Directory | Engine Code |
|----------|---------------------------|----------------------------|-------------|
| nlp | `vera-ai-nlp-reviewing/` | `vera-ai-nlp-generating/` | `vera-ai-nlp-generating/scripts/python/` |
| structured | `vera-ai-structured-reviewing/` | `vera-ai-structured-generating/` | `vera-ai-structured-generating/scripts/python/` |
| image | `vera-ai-image-reviewing/` | `vera-ai-image-generating/` | `vera-ai-image-generating/scripts/python/` |

## Models Implemented Per Modality

### NLP (`vera-ai-nlp-generating`)
- **Implemented now**: TF-IDF + Logistic Regression baseline
- **Implemented now**: SVM (linear + RBF), Random Forest, LightGBM
- **Implemented now**: GRU (bidirectional), TextCNN, ALBERT
- **Implemented optional variants**: tabular feature fusion and categorical embeddings
- **Not shipped in this open-source build**: BERT, RoBERTa, DeBERTa, prompt-only classifiers, SetFit

### Structured (`vera-ai-structured-generating`)
- **Implemented now**: LightGBM baseline
- **Implemented now**: Logistic Regression / Ridge, SVM, Random Forest, XGBoost, LightGBM, CatBoost
- **Implemented now**: MLP, TabNet, stacking, weighted voting
- **Tasks supported**: classification and regression

### Image (`vera-ai-image-generating`)
- **Implemented now**: Simple CNN or ResNet18 feature extractor baseline
- **Implemented now**: ResNet50, EfficientNet-B0, VGG16, DenseNet121
- **Implemented now**: ViT-B/16, weighted voting, stacking
- **Implemented now**: GradCAM and ViT attention maps
- **Planned but not shipped**: ConvNeXt-Tiny

## Workflow Files Per Skill

Reviewing skills contain:

```text
workflow/step01-collect-inputs.md
workflow/step02-check-distribution.md
workflow/step03-run-primary-test.md
```

Generating skills contain:

```text
workflow/step04-run-additional-models.md
workflow/step05-analyze-subgroups.md
workflow/step06-fit-advanced-models.md
workflow/step07-compare-models.md
workflow/step08-generate-manuscript.md
scripts/python/
reference/
```

## How the Pipeline Uses This Table

1. Step 02 confirms the data modality.
2. Look up the matching reviewing and generating skill paths.
3. Step 04 dispatches tracks: T1 reads steps 01-03 from the reviewing skill; T2-T5 read steps 04-08 from the generating skill.
4. Each track imports reusable modules from the generating skill's `scripts/python/` directory.
5. Output variation references are read from the generating skill's `reference/` directory.

## Modality-Specific Track Defaults

See `method-tracks.md` for canonical T1-T5 definitions per modality.
