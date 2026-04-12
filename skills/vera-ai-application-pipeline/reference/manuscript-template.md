# Manuscript Template

Markdown structure for the assembled manuscript. All manuscripts follow this section order regardless of modality or domain.

## Template

```markdown
# [Title]

## Abstract

[150-250 words. Written LAST after all other sections.]

**Background**: [1-2 sentences: context + gap]
**Objective**: [1 sentence: what this study does]
**Methods**: [2-3 sentences: data, models, evaluation protocol]
**Results**: [3-4 sentences: main findings with key numbers]
**Conclusions**: [1-2 sentences: implications]

**Keywords**: [3-5 keywords, separated by commas]

---

## 1. Introduction

[3-5 paragraphs, 600-1000 words]

[Para 1: Broad context and significance]
[Para 2: What is known -- prior approaches from literature]
[Para 3: Gap -- what remains unknown, why existing approaches fall short]
[Para 4: This study -- objective, approach, hypothesis]
[Para 5 (optional): Contribution and paper outline]

## 2. Related Work

### 2.1 [Theme 1: e.g., Pre-trained Models for Task X]

[Summary of approaches, key results, limitations]

### 2.2 [Theme 2: e.g., Classical ML Approaches]

[How traditional methods have been applied]

### 2.3 [Theme 3: e.g., Domain-Specific Methods]

[Specialized approaches for this domain]

[Position our work: "Unlike prior work that focuses on X, we Y."]

## 3. Data and Experimental Setup

### 3.1 Data Description

[Dataset source, collection, size, modality, preprocessing]

### 3.2 Features / Input Representation

[Tokenization, transforms, feature engineering, embedding]

### 3.3 Experimental Protocol

[Train/val/test splits, metrics, seeds, hardware]

## 4. Methods

[From merged methods.md -- reordered to standard flow]

### 4.1 Problem Formulation
[Formal task definition]

### 4.2 Baseline Models
[Classical ML + standard DL baselines]

### 4.3 Proposed / Main Models
[Architecture, training strategy, loss]

### 4.4 Training Details
[Optimizer, scheduler, augmentation, regularization]

### 4.5 Ensemble / Aggregation
[If applicable]

### 4.6 Interpretability / Analysis
[Error analysis, feature importance, attention visualization]

## 5. Results

[From merged results.md -- follows Section 4 ordering]

### 5.1 Main Results
[Model comparison table, key metrics]

### 5.2 Ablation Study
[Component contribution analysis]

### 5.3 Analysis
[Error analysis, per-class performance, failure cases]

### 5.4 Efficiency
[Params, FLOPs, training time comparison]

## 6. Discussion

[5-7 paragraphs, 800-1200 words]

[Para 1: Key findings in plain language]
[Para 2-3: Comparison with prior work]
[Para 4: Methodological strengths -- multi-model value]
[Para 5: Limitations (>= 3)]
[Para 6: Implications and future directions]

## References

[From merged references.bib -- formatted per venue style]

## Tables

**Table 1**: Dataset Statistics
[Split sizes, class distribution, feature counts]

**Table 2**: Main Results
[All models, all metrics, mean +/- std, bold best]

**Table 3**: Ablation Study
[Component removal analysis]

...

**Table N**: Model Efficiency Comparison
[Params, FLOPs, training time, inference speed]

## Figure Captions

**Figure 1**: [Architecture diagram / Data pipeline]
**Figure 2**: [Training curves -- loss + metric vs. epoch]
**Figure 3**: [Main results visualization]
...
**Figure N**: [Error analysis / Attention visualization]

[Actual figures stored in output/figures/ as PNG files]
```

## Section Length Guidelines

| Section | Words | Pages (approx) |
|---------|-------|-----------------|
| Abstract | 150-250 | -- |
| Introduction | 600-1000 | 1-2 |
| Related Work | 500-800 | 1-2 |
| Data & Setup | 400-700 | 1-2 |
| Methods | 600-1200 | 2-3 |
| Results | 800-1500 | 2-4 |
| Discussion | 800-1200 | 2-3 |
| **Total** | **3850-6650** | **9-16** |

## Adaptation Notes

- **NeurIPS/ICML/ICLR**: 9 pages + unlimited refs/appendix. Methods-heavy, compact related work.
- **ACL/EMNLP**: 8 pages + refs. Strong related work section expected. Error analysis valued.
- **CVPR/ECCV**: Results-heavy with many figures. Qualitative examples expected.
- **AAAI**: 7 pages. Concise throughout.
- **JMLR/TMLR**: No page limit. Can be thorough on all sections.
- **Domain venues (bioinformatics, medical imaging, etc.)**: Background section more extensive; methods may be shorter if using standard architectures.
