# 07 --- Compare Models

## Executor: Main Agent (code generation)

## Data In: All model outputs from steps 03-06

## Design Principle

Models are analytic lenses, not contestants. Each architecture captures different
visual patterns in image data. The comparison step synthesizes what converges and what
each uniquely reveals. Never frame as "which model is best." No F1 horse race.

## Generate code for PART 6: Cross-Method Insight Synthesis

### 7A: Unified Performance Table

| Model | F1 (weighted) | 95% CI | AUC (macro) | 95% CI | Params (M) | FLOPs (G) | Notes |
|-------|---------------|--------|-------------|--------|------------|-----------|-------|
| Baseline CNN | ... | ... | ... | ... | ... | ... | From testing |
| ResNet50 | ... | ... | ... | ... | 25.6 | 4.1 | Feature extraction + fine-tuning |
| EfficientNet-B0 | ... | ... | ... | ... | 5.3 | 0.4 | Compound scaling |
| VGG16 | ... | ... | ... | ... | 138.4 | 15.5 | Deep feature hierarchy |
| DenseNet121 | ... | ... | ... | ... | 8.0 | 2.9 | Dense connections |
| ViT-B/16 | ... | ... | ... | ... | 86.6 | 17.6 | Patch-based transformer |
| Ensemble | ... | ... | ... | ... | - | - | Soft voting / stacking |

### 7B: Unified Feature Attribution Table (GradCAM + Attention)

Normalize all attribution measures to a 0-100 scale (max = 100):
- ResNet50: GradCAM activation rescaled
- EfficientNet-B0: GradCAM activation rescaled
- VGG16: GradCAM activation rescaled
- DenseNet121: GradCAM activation rescaled
- ViT: attention map intensity rescaled

| Spatial Region | ResNet50 (GradCAM) | EfficientNet (GradCAM) | VGG16 (GradCAM) | DenseNet (GradCAM) | ViT (attention) | Consensus |
|----------------|--------------------|-----------------------|------------------|--------------------|--------------------|-----------|

Consensus = agreement across methods on most attended regions.

### 7C: Insight Synthesis Table

| Method Family | Unique Insight |
|---------------|----------------|
| Deep Residual (ResNet50) | [skip connections, gradient flow, deep feature hierarchies] |
| Efficient Scaling (EfficientNet-B0) | [compound scaling, mobile-friendly, efficiency-accuracy tradeoff] |
| Deep Sequential (VGG16) | [simple deep architecture, large receptive field] |
| Dense Connection (DenseNet121) | [feature reuse, parameter efficiency, medical imaging suitability] |
| Patch Transformer (ViT) | [global self-attention, long-range dependencies, patch-level reasoning] |
| Ensemble | [model diversity, complementary error patterns, robustness] |

### 7D: Narrative Synthesis

3-4 sentences covering:
1. What converges across methods (strongest discriminative regions agreement via GradCAM/attention)
2. What CNN architectures uniquely reveal (local texture, spatial hierarchies)
3. What ViT uniquely reveals (global context, patch-level attention patterns)
4. Overall: convergence of spatial attribution strengthens confidence in key finding

### Quality: Synthesis variation
Apply sentence bank (model comparison section). Rotate lead-in.

## Validation Checkpoint

- [ ] Unified performance table with all 6+ models including param counts and FLOPs
- [ ] Unified attribution table with GradCAM + attention on 0-100 scale
- [ ] Region consensus computed
- [ ] Insight synthesis table: one row per architecture family
- [ ] No F1 horse-race framing
- [ ] Narrative synthesis: 3-4 sentences
- [ ] Sentence bank applied

## Data Out -> 08-generate-manuscript.md

```
comparison_code_py: [PART 6 Python code]
unified_performance_table: [all models with params/FLOPs]
unified_attribution_table: [GradCAM + attention, 0-100]
insight_table: [architecture family x unique insight]
results_para_comparison: [synthesis paragraph prose]
```
