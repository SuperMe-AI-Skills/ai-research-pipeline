# Manuscript Writing: LaTeX for ML Venues

Write LaTeX manuscript sections for AI/ML conference or journal submission.

## Input

- `PAPER_PLAN.md` -- section outline, claims-evidence matrix
- `paper/figures/` -- all figures ready
- `output/` or root-level results files
- Target venue from PIPELINE_STATE.json

## Venue Templates

| Venue | Style File | Page Limit | Key Notes |
|-------|-----------|------------|-----------|
| NeurIPS | neurips_2026.sty | 9 + refs + appendix | Single column, 10pt |
| ICML | icml2026.sty | 9 + refs + appendix | Two column abstract |
| ICLR | iclr2026_conference.sty | 9 + refs + appendix | OpenReview format |
| ACL | acl.sty | 8 + refs | Long paper; short = 4 |
| CVPR | cvpr.sty | 8 + refs + supp | Two column |
| AAAI | aaai26.sty | 7 + refs | Two column |
| JMLR | jmlr.sty | No limit | Journal format |
| TMLR | tmlr.sty | No limit | OpenReview journal |
| General | article.cls | Flexible | 11pt, single column |

## Section Writing Guidelines

### Abstract
- 150-250 words
- Structure: problem, approach (1 sentence), key result (with numbers), implication
- No citations, no abbreviations (unless universal)
- Must stand alone

### Introduction
- Start with broad context, narrow to specific problem
- Clearly state the gap
- "In this paper, we propose..." -- be direct
- End with contribution list:
  ```latex
  Our contributions are as follows:
  \begin{itemize}
  \item We propose [method], which [key innovation].
  \item We demonstrate [key result] on [benchmarks].
  \item We provide [analysis/insight].
  \end{itemize}
  ```

### Related Work
- Organize by theme, not chronologically
- End each paragraph with differentiation: "Unlike [prior work], our approach..."
- Be fair to related work -- acknowledge their strengths

### Method
- Start with problem formulation (mathematical notation)
- Build up from simple to complex
- Use `\newcommand` for model name and recurring notation
- Include algorithm box if method has clear steps:
  ```latex
  \begin{algorithm}
  \caption{Training procedure for \ours}
  ...
  \end{algorithm}
  ```

### Experiments
- Setup subsection FIRST: datasets, baselines, metrics, implementation details
- Main results table with bold best, underline second best
- Ablation study table
- Analysis subsection (error analysis, qualitative examples)
- All in paper/sections/results.tex

### Conclusion
- Summarize contributions (not results -- those are in abstract)
- Limitations (be honest)
- Future work (2-3 directions)
- Do NOT introduce new results

## LaTeX Conventions

```latex
% Custom commands (in preamble)
\newcommand{\ours}{\textsc{MethodName}}
\newcommand{\dataset}[1]{\textsc{#1}}
\newcommand{\metric}[1]{\text{#1}}

% Citations
\citet{smith2024} showed that...         % Smith et al. (2024) showed...
...as shown in prior work~\citep{jones2023}.  % ...(Jones et al., 2023).

% Tables
\begin{table}[t]
\caption{Main results. \textbf{Bold} = best, \underline{underline} = second.}
\centering
\small
\begin{tabular}{lccc}
\toprule
Method & Dataset 1 & Dataset 2 & Avg \\
\midrule
Baseline & $80.1_{\pm 0.3}$ & ... & ... \\
\ours & $\mathbf{82.3}_{\pm 0.4}$ & ... & ... \\
\bottomrule
\end{tabular}
\end{table}

% Figures
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig1.pdf}
\caption{Architecture overview of \ours.}
\label{fig:arch}
\end{figure}
```

## Writing Quality Standards

| Rule | Example |
|------|---------|
| Active voice preferred | "We propose" not "It is proposed" |
| Specific claims | "improves by 2.3%" not "significantly improves" |
| Quantify everything | "on 3 datasets" not "on multiple datasets" |
| No AI-isms | Avoid "delve", "leverage", "harness", "paradigm shift" |
| Consistent tense | Present for method, past for experiments |
| Define before use | Define all notation on first use |
| Cross-reference | "\autoref{tab:main}" or "Table~\ref{tab:main}" |

## Output

- `paper/main.tex` -- master document
- `paper/sections/abstract.tex`
- `paper/sections/introduction.tex`
- `paper/sections/related_work.tex`
- `paper/sections/methods.tex`
- `paper/sections/results.tex`
- `paper/sections/discussion.tex` (or conclusion.tex)
- `paper/sections/appendix.tex`
- `paper/references.bib`

## Key Rules

- Stay within page limits (check after compilation)
- Every claim in the paper must have evidence in a table/figure
- Bold best results, underline second best
- Report mean +/- std for all multi-seed results
- Include implementation details in appendix if space-constrained
- Use `\footnote` sparingly
