# 08 --- Generate Manuscript

## Executor: Main Agent (assembly)

## Data In: All code, prose, tables, plots, and style_vector from steps 04-07

## Assemble methods.md

### Structure (order fixed, content varies)

```markdown
## Methods

### Text Preprocessing
[Para 1: Text cleaning steps, tokenization approach]

### Feature Engineering
[Para 2: TF-IDF specification, extra features if used]

### Machine Learning Models
[Para 3: LogReg specification + hyperparameter search strategy]
[Para 4: SVM, RF, LightGBM specification]

### Deep Learning Models
[Para 5: GRU architecture, vocabulary, padding, training strategy]
[Para 6: TextCNN architecture, filter sizes, training strategy]
[Para 7: ALBERT fine-tuning, pre-trained model, tabular fusion if applicable]

### Evaluation
[Para 8: Metrics (F1, AUC), bootstrapped CIs, train/val/test protocol]

### Model Comparison
[Para 9: Unified importance normalization, cross-method synthesis approach]

### Software
[Para 10: Python version, key packages with versions]
```

### Rules
- Write as if a human analyst chose these methods for THIS specific study
- Never expose pipeline logic or decision rules
- State what was done + why + key parameters
- No results in methods; no code in methods
- Cite methodological references where appropriate
- Follow `reference/rules/reporting-standards.md`

### Quality: Methods variation
- Each paragraph selects one framing from 3 options
- Vary passive vs active voice across paragraphs
- Vary whether hyperparameter ranges are stated or summarized

## Assemble results.md

### Section ordering logic

Choose ONE ordering based on research question emphasis:
- **Order A (benchmark-driven):** Baseline → ML models → DL models → Comparison
- **Order B (architecture-driven):** ALBERT → GRU → TextCNN → ML models → Comparison
- **Order C (interpretability-driven):** LogReg → Importance → ML ensemble → DL → Comparison

### Rules
- All metrics are final computed values
- Apply sentence bank rotation
- Include 1-2 cross-references between sections
- Tables and figures referenced by number
- Follow `reference/rules/reporting-standards.md`

## Generate references.bib

- Include ONLY references actually cited
- BibTeX format
- Include: methodological, software, reporting guideline citations

## Apply code style variation

Apply style_vector to final code.py per `reference/specs/code-style-variation.md`.

## Validation Checkpoint

- [ ] methods.md contains no results or numbers
- [ ] methods.md covers all analysis steps
- [ ] results.md section order matches research question type
- [ ] All numbers are computed, no placeholders
- [ ] Table/figure references match actual files
- [ ] Metric formatting follows reporting-standards.md
- [ ] F1/AUC with 95% CIs throughout
- [ ] references.bib includes all cited works only
- [ ] Code style variation applied to final code.py
- [ ] No meta-commentary about pipeline structure

## Data Out -> Final Deliverables

```
Deliverables:
├── nlp_analysis.py              (PARTS 0-6, style-varied)
├── methods.md                    (manuscript Methods section)
├── results.md                    (manuscript Results section)
├── references.bib                (cited works only)
├── tables/
│   ├── performance_table.csv
│   ├── importance_table.csv
│   └── comparison_table.csv
└── figures/
    ├── plot_01_data_overview.png
    ├── plot_02_confusion_roc.png
    ├── plot_03_*.png             (ML model plots)
    ├── plot_04_subgroup_*.png    (if applicable)
    ├── plot_05_*.png             (DL model plots)
    └── plot_06_comparison.png
```
