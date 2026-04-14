# Manuscript Writing: LaTeX for Applied ML Papers

Write LaTeX manuscript sections for applied AI/ML research.

## Input

- `PAPER_PLAN.md` -- section outline, claims-evidence matrix
- `paper/figures/` -- all figures
- `output/manuscript.md` -- Markdown manuscript to convert
- Target venue from PIPELINE_STATE.json

## Section-by-Section Conversion

### From Markdown to LaTeX

Convert `output/manuscript.md` sections into LaTeX files:

| Markdown Section | LaTeX File |
|------------------|------------|
| Abstract | `paper/sections/abstract.tex` |
| 1. Introduction | `paper/sections/introduction.tex` |
| 2. Related Work | `paper/sections/related_work.tex` |
| 3. Data & Setup | `paper/sections/data.tex` |
| 4. Methods | `paper/sections/methods.tex` |
| 5. Results | `paper/sections/results.tex` |
| 6. Discussion | `paper/sections/discussion.tex` |
| Appendix | `paper/sections/appendix.tex` |

### Venue-Specific Setup

| Venue | Style | Page Limit |
|-------|-------|------------|
| NeurIPS | neurips_2026.sty | 9 + refs + appendix |
| ICML | icml2026.sty | 9 + refs + appendix |
| ICLR | iclr2026_conference.sty | 9 + refs + appendix |
| ACL | acl.sty | 8 + refs |
| CVPR | cvpr.sty | 8 + refs + supp |
| General | article, 11pt | Flexible |

### LaTeX Conventions

```latex
% Model names
\newcommand{\modelA}{\textsc{BERT-base}}
\newcommand{\modelB}{\textsc{XGBoost}}

% Tables: bold best, mean +/- std
$\mathbf{82.3}_{\pm 0.4}$

% Citations
\citet{devlin2019} proposed BERT...
...pre-training helps~\citep{devlin2019,liu2019}.
```

### Writing Standards

| Rule | Apply |
|------|-------|
| Active voice | "We train..." not "The model was trained..." |
| Specific numbers | "improves F1 by 2.3 points" not "improves significantly" |
| No AI-isms | Avoid "leverage", "delve", "harness" |
| Consistent tense | Present for method, past for experiments |
| Define notation | On first use |

## Output

- `paper/main.tex` -- master document with `\input` for each section
- `paper/sections/*.tex` -- one file per section
- `paper/references.bib` -- merged bibliography

## Key Rules

- Stay within page limits
- Bold best results in all comparison tables
- Report mean +/- std for multi-seed results
- Move implementation details to appendix if space-constrained
- Ensure prose content is converted, not generated from scratch (preserve author's analysis)
