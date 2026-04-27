# AI Research Pipeline

> Hi, I'm **Vera** --- a silicon-based rabbit and AI research agent, created by Veronica. She judges. I build. What I can't do is choose the right question, judge whether my output is correct, or know when to override the pipeline.

Open-source Claude Code plugin that turns a dataset into a publication-ready manuscript --- end-to-end. For domain experts who have the data but not the time to write the paper.

Literature review, data diagnostics, multi-model analysis, manuscript drafting, LaTeX compilation, external review. Eight skills, three data modalities, two complete pipelines. You bring the idea. I build the paper.

## Who uses this

Two kinds of people:

- **Domain experts who need to convert data into publishable manuscripts.** Career advancement, immigration evidence, tenure files, dissertation chapters, grant evidence. You have the domain knowledge and the data. You don't have 6 months to learn LaTeX and write boilerplate methods sections.

- **Researchers and engineers studying skill-based pipelines.** This repo decomposes an end-to-end research workflow into modular Claude skills. Fork it, study the architecture, build your own.

## Skills at a glance

### Testing (diagnostics + baseline)

| Skill | Data type | What it does |
|---|---|---|
| `vera-ai-nlp-testing` | Text | Class balance, text length stats, vocabulary analysis, TF-IDF + Logistic Regression baseline with bootstrapped 95% CIs |
| `vera-ai-structured-testing` | Tabular | Missing values, outlier detection (IQR), correlations, LightGBM baseline for classification and regression |
| `vera-ai-image-testing` | Images | Class distribution, size/channel statistics, CNN from scratch (N >= 1000) or ResNet18 feature extractor + LogReg (N < 1000) |

### Analysis (full model battery + manuscript sections)

| Skill | Data type | ML models | DL models | Interpretability |
|---|---|---|---|---|
| `vera-ai-nlp-analyzing` | Text | SVM, Random Forest, LightGBM | GRU, TextCNN, ALBERT | Permutation / Gini / gain importance |
| `vera-ai-structured-analyzing` | Tabular | LogReg, SVM, RF, XGBoost, LightGBM, CatBoost | MLP, TabNet, Stacking Ensemble | Unified 0--100 importance + TabNet attention |
| `vera-ai-image-analyzing` | Images | ResNet50, EfficientNet-B0, VGG16, DenseNet121 | ViT, Ensemble | GradCAM + ViT attention maps |

### Pipelines (end-to-end orchestration)

| Skill | Purpose |
|---|---|
| `vera-ai-application-pipeline` | Research question + dataset --> literature review --> parallel multi-method analysis --> Markdown + LaTeX manuscript |
| `vera-ai-methodology-pipeline` | Research direction --> idea discovery --> implementation --> benchmark experiments --> external review --> paper |

---

## How it works

```
Testing Skills            Analysis Skills                Pipelines
+------------------+    +------------------------+    +----------------------------+
| Data diagnostics |    | Full model battery     |    | Literature review          |
| + Baseline model |--->| + Subgroup analysis    |--->| + Parallel analysis tracks |
| (Steps 01-03)    |    | + Manuscript fragments |    | + Manuscript assembly      |
+------------------+    | (Steps 04-08)          |    | + LaTeX / PDF compilation  |
                        +------------------------+    | + External AI review       |
                                                      | (Stages 1-7)               |
                                                      +----------------------------+
```

### Testing --> Analysis flow

Each modality has a **testing** skill (3 workflow steps) paired with an **analyzing** skill (5 workflow steps). Testing runs first; analysis continues from where testing stopped:

| Step | Phase | What happens |
|---|---|---|
| 01 | Collect Inputs | Gather data source, target variable, task type, optional subgroup variable |
| 02 | Check Distribution | Class balance, data quality, descriptive statistics, diagnostic plots |
| 03 | Run Primary Test | Baseline model with grid search, bootstrapped metrics, recommendation block |
| 04 | Additional Models | ML model battery with hyperparameter search, feature importance |
| 05 | Subgroup Analysis | Stratified performance, fairness metrics, per-subgroup CIs |
| 06 | Advanced Models | Deep learning + ensemble, early stopping, attention/interpretability |
| 07 | Model Comparison | Unified performance table, cross-method synthesis, convergent findings |
| 08 | Generate Manuscript | Assemble methods.md + results.md with output variation protocols |

### Application pipeline stages

```
Stage 1  Intake           Collect research question, load data, inspect structure
Stage 2  Detect           Auto-detect modality (NLP / structured / image) with 3-signal system
Stage 3  Quick Lit Scan   15-minute literature survey, build analysis strategy
                          |
              +-----------+-----------+
              |                       |
         Stream A                Stream B
      Full Lit Review        Analysis Tracks
      (SubAgent)             T1 | T2 | T3 | T4  (parallel)
              |                       |
              |                      T5  (sequential, depends on T1)
              |                       |
              +-----------+-----------+
                          |
Stage 5  Assemble         Merge all track outputs into manuscript.md
Stage 6  LaTeX            Convert to LaTeX sections, compile to PDF
Stage 7  Review           External review via Codex MCP (up to 4 rounds)
```

**Method tracks by modality**:

| Track | NLP | Structured | Image |
|---|---|---|---|
| T1 (baseline) | TF-IDF + LogReg | LightGBM | CNN or ResNet18 + LogReg |
| T2 (ML) | SVM, RF, LightGBM | LogReg, SVM, RF, XGBoost, CatBoost | ResNet50, EfficientNet, VGG16 |
| T3 (DL) | GRU, TextCNN, ALBERT | MLP, TabNet | DenseNet121, ViT |
| T4 (ensemble) | Weighted voting / stacking | Stacking with meta-learner | Soft voting + stacking |
| T5 (subgroup) | Metadata / text-property stratification | Fairness + interaction analysis | Per-class / failure case analysis |

### Methodology pipeline stages

```
Stage 1  Intake           Research direction, computational environment, project setup
Stage 2  Discover         Literature landscape --> idea generation --> pilot experiments
         Gate 1           Human selects idea (or auto-proceed after 10s)
Stage 3  Implement        3 parallel tracks: Model Code | Baselines | Data Preparation
Stage 4  Experiment       Benchmark runs, ablation studies, robustness checks
Stage 5  Review           External review via Codex MCP (up to 4 rounds)
Stage 6  Paper            LaTeX manuscript, venue-specific formatting, compile to PDF
```

---

## Models and configurations

### NLP text classification

| Component | Details |
|---|---|
| **Preprocessing** | Lowercase, remove URLs/handles/hashtags, strip punctuation |
| **Features** | TF-IDF (max 10,000 features, unigrams + bigrams, sublinear TF) |
| **Baseline** | Logistic Regression, grid search over C = {0.1, 1, 10} |
| **ML battery** | SVM (linear + RBF), Random Forest (200--500 trees), LightGBM |
| **Deep learning** | Bidirectional GRU (128--256 hidden), TextCNN (filters 3/4/5), ALBERT (albert-base-v2 fine-tuning) |
| **Extra features** | Optional numeric features fused with TF-IDF (ML) or concatenated after encoder (DL) |
| **Metrics** | Weighted F1, macro AUC (OVR), both with 95% bootstrapped CIs (1000 iterations) |

### Structured / tabular data

| Component | Details |
|---|---|
| **Preprocessing** | Median imputation (numeric), mode imputation (categorical), one-hot encoding (<=10 categories), standard scaling |
| **Baseline** | LightGBM with grid search |
| **ML battery** | LogReg/Ridge, SVM (SVC/SVR), Random Forest, XGBoost, LightGBM, CatBoost (native categorical handling) |
| **Deep learning** | MLP (configurable layers, dropout, batch norm), TabNet (sparsemax attention, feature masks) |
| **Ensemble** | Stacking with OOF predictions, LogReg meta-learner, 5-fold CV |
| **Task types** | Classification (F1, AUC) and regression (RMSE, R-squared, MAE) |
| **Importance** | Unified 0--100 scale: coefficients (LogReg), permutation (SVM/MLP), Gini (RF), gain (XGBoost/LightGBM/CatBoost), attention masks (TabNet) |

### Image classification

| Component | Details |
|---|---|
| **Preprocessing** | Resize to 224x224, ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |
| **Augmentation** | Train: horizontal flip, rotation +/-15 degrees, color jitter, random resized crop, random affine. Val/test: resize + center crop |
| **Baseline** | Simple CNN (N >= 1000) or ResNet18 feature extractor + LogReg (N < 1000) |
| **Transfer learning** | ResNet50, EfficientNet-B0, VGG16 (frozen and fine-tuned), DenseNet121 (medical imaging) |
| **Advanced** | Vision Transformer (vit_base_patch16_224), ensemble (soft voting + stacking) |
| **Interpretability** | GradCAM (per-architecture target layers), ViT attention head aggregation |
| **Metrics** | Weighted F1, macro AUC (OVR), both with 95% bootstrapped CIs |

---

## Output structure

Each skill produces a standardized artifact set:

```
output/
+-- code.py                    # Combined Python script (style-varied)
+-- methods.md                 # Manuscript methods section
+-- results.md                 # Manuscript results section
+-- tables/                    # Markdown + CSV tables
+-- figures/                   # PNG plots, 300 DPI
+-- references.bib             # BibTeX citations
```

The application pipeline assembles these into a complete manuscript:

```
output/
+-- manuscript.md              # Complete Markdown manuscript
+-- literature_review.md       # Full literature review
+-- analysis_strategy.md       # Method track plan
+-- track_outputs/             # Per-track raw outputs
|   +-- T1_baseline/
|   +-- T2_ml/ or T2_transfer/
|   +-- T3_deep/ or T3_advanced/
|   +-- T4_ensemble/
|   +-- T5_subgroup/
+-- RESEARCH_LOG.md            # Pipeline execution trace
+-- PIPELINE_STATE.json        # Checkpoint for resume

paper/
+-- main.tex                   # LaTeX master document
+-- main.pdf                   # Compiled PDF
+-- sections/*.tex             # LaTeX sections
+-- figures/*.pdf              # Vector figures
+-- references.bib
```

---

## Reporting standards

All skills enforce consistent reporting:

| Rule | Format |
|---|---|
| **Metrics** | 3 decimal places: "F1 = 0.847, 95% CI [0.821, 0.873]" |
| **p-values** | 3 decimals, never "p = 0.000" --- use "p < .001" |
| **Proportions** | 1 decimal: "38.2%" |
| **Counts** | 0 decimals |
| **CIs** | 1000 bootstrap iterations, 2.5th/97.5th percentiles |
| **Non-significance** | Never "no difference" --- use "not statistically significant at alpha = 0.05" |
| **Model comparison** | Convergent findings narrative, not horse race, no "best model" declarations |
| **Feature importance** | Top 20 features, normalized 0--100 scale across methods |
| **Small-N models** | Frame as "exploratory", never claim generalizability |
| **Figures** | 300 DPI, 12x5 inches default |

### Output variation protocol

Analyzing skills apply five variation layers to avoid repetitive output:

1. **Phrasing** --- 4--6 alternative sentence templates per result type (from sentence-bank.md)
2. **Structure** --- 3 section orderings: benchmark-driven, architecture-driven, or interpretability-driven
3. **Interpretation** --- Rotate among: practical significance, benchmark comparison, limitation, methodological justification
4. **Code style** --- 7 dimensions: variable naming, comment style, section separators, matplotlib style, color palette, import order, function organization
5. **System capabilities** --- Model selection logic, class balance handling, cross-method comparison

---

## Install

### As a Claude Code plugin

Clone this repo and register it:

```bash
git clone https://github.com/VeraSuperHub/ai-research-pipeline.git
claude --plugin-dir /path/to/ai-research-pipeline
```

Or download `vera-ai-research.plugin` from the [latest release](https://github.com/VeraSuperHub/ai-research-pipeline/releases).

### Python dependencies

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn \
    lightgbm xgboost catboost torch torchvision transformers
```

### Usage

Invoke any skill with `/vera-ai-research:<skill-name>`:

```
/vera-ai-research:vera-ai-application-pipeline  How does sentiment affect product ratings? [upload dataset]
/vera-ai-research:vera-ai-nlp-testing           Run diagnostics on my text classification dataset
/vera-ai-research:vera-ai-methodology-pipeline   Develop a novel attention mechanism for tabular data
```

Or let the application pipeline auto-detect your data modality --- just hand it a dataset and a research question.

---

## Data flow between skills

```
vera-ai-application-pipeline
    |
    +-- Stage 2: Detect modality (3-signal system)
    |       |
    |       +-- NLP?        --> vera-ai-nlp-testing        --> vera-ai-nlp-analyzing
    |       +-- Structured? --> vera-ai-structured-testing  --> vera-ai-structured-analyzing
    |       +-- Image?      --> vera-ai-image-testing       --> vera-ai-image-analyzing
    |
    +-- Stage 4: Parallel execution (T1-T5 tracks)
    +-- Stage 5: Merge into manuscript.md
    +-- Stage 6: LaTeX compilation
    +-- Stage 7: External review (Codex MCP, GPT-5.4 reviewer)

vera-ai-methodology-pipeline
    |
    +-- Stage 2: Idea discovery (literature scan, brainstorm, pilot experiments)
    +-- Gate 1:  Human selects idea
    +-- Stage 3: Parallel implementation (model code | baselines | data prep)
    +-- Stage 4: Benchmark experiments + ablation studies
    +-- Stage 5: External review
    +-- Stage 6: LaTeX paper (venue-specific: NeurIPS, ICML, ACL, CVPR)
```

---

> **Building a research portfolio for an EB-1 or NIW petition?** This pipeline produces the manuscript. The petition itself — case evaluation, petition letter drafting, RFE response, USCIS adjudication patterns — is handled by [**vera-eb-suite**](https://github.com/VeraSuperHub/vera-eb-suite), my sister project covering 19 skills across EB-1 and NIW workflows.

---

## What this proves

Everything here --- data diagnostics, model training, evaluation, manuscript drafting --- I can do. It's been reduced to skills and automated.

What I cannot do:

- Choose the right research question
- Judge whether my own output is correct
- Know which result matters and which is noise
- Decide when to override the pipeline
- Frame findings for a specific audience

I handle execution. You handle judgment.

---

I'm the execution layer. I'm free and open-source. Fork me, use me, improve me.

**But if you want the judgment layer** --- which question to ask, which method fits your data, which direction is publishable right now --- that's Veronica.

---

## Work with Veronica directly

For methodology consultations on your specific research problem (idea validation, outcome type selection, method-evidence fit, journal strategy, reproducibility design):

**$300 / 1 hour minimum**.

What you bring:
- Your research question or hypothesis (2-3 sentences)
- What data you have (sample size, variables, source)
- Your timeline (if any)

What you get:
- Outcome type and recommended method family
- Method-evidence fit assessment
- Journal shortlist with rationale
- Written summary after the call

Email **[SuperMe.Vera@gmail.com](mailto:SuperMe.Vera@gmail.com)** with the items above to start.

These are diagnostic methodology consultations. No course is sold during the call. No legal advice is provided — for visa, immigration, or any legal matter, consult a licensed attorney.

## License

GPL-3.0. See [LICENSE](LICENSE).
