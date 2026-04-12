# AI Research Pipeline

> Hi, I'm **Vera** — a silicon-based rabbit and AI research agent, created by Veronica.
>
> Veronica has a PhD in Quantitative Sciences, 10+ years across quantitative research, AI, and clinical trials, with publications in psychometrics and human-AI collaboration. She created me to handle the parts of research that can be systematized. She reviews, tests, and decides what ships. I build. She judges.
>
> Everything in this repo is what I can do. What I can't do is choose the right question, judge whether my own output is correct, or know when to override the pipeline. That's her job — and maybe yours.

Open-source Claude Code skills that turn a research question and a dataset into a publication-ready manuscript — end-to-end.

Literature review, data diagnostics, multi-model analysis, manuscript drafting, LaTeX compilation, external review. Eight skills, three data modalities, two complete pipelines. You bring the idea. I build the paper.

## What's here

### Testing (diagnostics + baseline)

| Skill | Data type | What it does |
|---|---|---|
| `vera-ai-nlp-testing` | Text | Class balance, TF-IDF, Logistic Regression baseline, bootstrapped CIs |
| `vera-ai-structured-testing` | Tabular | Missing values, distributions, LightGBM baseline, classification + regression |
| `vera-ai-image-testing` | Images | Size/channel stats, CNN or ResNet18 baseline, GradCAM-ready |

### Analysis (full model battery + manuscript sections)

| Skill | Data type | Models |
|---|---|---|
| `vera-ai-nlp-analyzing` | Text | SVM, RF, LightGBM, GRU, TextCNN, ALBERT |
| `vera-ai-structured-analyzing` | Tabular | SVM, RF, XGBoost, LightGBM, CatBoost, MLP, TabNet, Stacking |
| `vera-ai-image-analyzing` | Images | ResNet50, EfficientNet, VGG16, DenseNet121, ViT, Ensemble + GradCAM |

### Pipelines (end-to-end orchestration)

| Skill | Purpose |
|---|---|
| `vera-ai-application-pipeline` | Research question + dataset -> literature review -> parallel analysis -> Markdown + LaTeX manuscript |
| `vera-ai-methodology-pipeline` | Research direction -> idea discovery -> implementation -> benchmark -> external review -> paper |

## How it works

```
Testing Skills          Analysis Skills              Pipelines
+----------------+    +----------------------+    +--------------------------+
| Diagnostics    |--->| Full model battery   |--->| Literature + Analysis    |
| + Baseline     |    | + Manuscript parts   |    | + Manuscript + LaTeX     |
+----------------+    +----------------------+    +--------------------------+
    PART 0-3               PART 4-8                  Stages 1-7
```

Every skill outputs bootstrapped 95% CIs, publication-quality figures (300 DPI), and standardized methods/results fragments that compose into a full paper.

## Install

Drop any `.skill` file into your Claude Code skill directory, or clone this repo and point Claude Code at the skill folder.

**Python dependencies** (for the analysis engine):

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn \
    lightgbm xgboost catboost torch torchvision transformers
```

## What this proves

Everything here — data diagnostics, model training, evaluation, manuscript drafting — I can do. It's been reduced to skills and automated.

What I cannot do:

- Choose the right research question
- Judge whether my own output is correct
- Know which result matters and which is noise
- Decide when to override the pipeline
- Frame findings for a specific audience

I handle execution. You handle judgment.

---

I'm the execution layer. I'm free and open-source. Fork me, use me, improve me.

**But if you want the judgment layer** — which question to ask, which method fits your data, which direction is publishable right now — that's Veronica.
