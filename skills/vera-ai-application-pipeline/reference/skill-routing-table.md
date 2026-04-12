# Skill Routing Table

Maps confirmed data modality to testing and analyzing skills.

All paths in this file are RELATIVE to a discovered `REPO_ROOT`. The pipeline
never hardcodes an absolute path. To resolve `REPO_ROOT` at runtime, follow
the discovery procedure documented in `workflow/step04-parallel.md` (section
4.1):

1. Use `$AIRESEARCH_ROOT` if set, OR
2. Derive it from the orchestrator skill's own absolute path (three levels up
   from `vera-ai-application-pipeline/workflow/step04-parallel.md`), OR
3. Walk up from `$PWD` until you find an ancestor that contains both testing
   and `vera-ai-analysis-engine/`.

`REPO_ROOT` should resolve to the AIResearch skill suite root directory.

## Routing Table

| Modality | Testing (steps 01-03) | Analysis (steps 04-08) | Engine Code |
|----------|---------------------------|----------------------------|-------------|
| nlp | `vera-ai-nlp-testing/` | `vera-ai-analysis-engine/nlp/vera-ai-nlp-analyzing/` | `.../nlp/vera-ai-nlp-analyzing/src/` |
| structured | `vera-ai-structured-testing/` | `vera-ai-analysis-engine/structured/vera-ai-structured-analyzing/` | `.../structured/vera-ai-structured-analyzing/src/` |
| image | `vera-ai-image-testing/` | `vera-ai-analysis-engine/image/vera-ai-image-analyzing/` | `.../image/vera-ai-image-analyzing/src/` |

## Models Implemented Per Modality

### NLP (`vera-ai-nlp-analyzing`)
- **Baseline**: TF-IDF + Logistic Regression
- **ML battery**: SVM (linear + RBF), Random Forest, LightGBM
- **Deep learning**: GRU (bidirectional), TextCNN (multi-filter), ALBERT (fine-tuned)
- **Optional**: tabular feature fusion (text + extras), categorical embeddings via `dl_*_extra.py`
- **Not implemented**: BERT, RoBERTa, DeBERTa, GPT/Claude few-shot, SetFit, prompt-based methods

### Structured (`vera-ai-structured-analyzing`)
- **Baseline**: LightGBM
- **ML battery**: Logistic Regression / Ridge, SVM (SVC/SVR), Random Forest, XGBoost, LightGBM, CatBoost
- **Deep learning**: MLP, TabNet (with sparsemax attention)
- **Ensemble**: Stacking (OOF + meta-learner), weighted voting (Nelder-Mead)
- **Tasks supported**: classification AND regression (F1/AUC and RMSE/R²/MAE)

### Image (`vera-ai-image-analyzing`)
- **Baseline**: Simple CNN (N>=1000) or ResNet18 feature extractor + LogReg (N<1000)
- **CNN architectures**: ResNet50, EfficientNet-B0, VGG16, DenseNet121
- **Transfer learning modes**: feature extraction (frozen backbone) + full fine-tuning
- **Transformer**: ViT (vit_b_16)
- **Ensemble**: Weighted voting + stacking
- **Interpretability**: GradCAM (CNN models), attention maps (ViT)

## Workflow Files Per Skill

### Testing Skills
```
vera-ai-{modality}-testing/
├── SKILL.md
├── config/default.json
├── workflow/
│   ├── 01-collect-inputs.md
│   ├── 02-check-distribution.md
│   └── 03-run-primary-test.md
└── reference/specs/citations.md
```

### Analyzing Skills
```
vera-ai-analysis-engine/{modality}/vera-ai-{modality}-analyzing/
├── SKILL.md
├── config/default.json
├── workflow/
│   ├── 04-run-additional-models.md
│   ├── 05-analyze-subgroups.md
│   ├── 06-fit-advanced-models.md
│   ├── 07-compare-models.md
│   └── 08-generate-manuscript.md
├── reference/
│   ├── specs/output-variation-protocol.md
│   ├── specs/code-style-variation.md
│   ├── rules/reporting-standards.md
│   └── patterns/sentence-bank.md
└── src/  ← reusable Python engine modules
```

**Important**: The analyzing skill does NOT contain workflow steps 01-03.
Those live in the paired testing skill. The pipeline must read both skills
to execute the full T1-T5 track battery.

## How the Pipeline Uses This Table

1. Step 02 confirms the data modality (nlp / structured / image)
2. Look up the matching testing + analyzing skill paths in this table
3. Step 04 dispatches tracks:
   - T1 reads workflow steps 01-03 from the testing skill
   - T2-T5 read workflow steps 04-08 from the analyzing skill
4. Each track imports models from the analyzing skill's `src/` engine
5. Output variation references are read from the analyzing skill's `reference/` directory

## Modality-Specific Track Defaults

See `method-tracks.md` for the canonical T1-T5 definitions per modality.
