# Step 06: LaTeX Manuscript & PDF Compilation

> **Executor**: Main Agent (invokes paper-writing sub-skills)
> **Input**: `output/manuscript.md` + all `output/` artifacts
> **Output**: `paper/main.tex`, `paper/sections/*.tex`, `paper/figures/*.pdf`, `paper/main.pdf`

---

## Execution Instructions

### 6.1 Prerequisites

Verify Stage 5 outputs exist:
- `output/manuscript.md` -- complete Markdown manuscript
- `output/figures/` -- all figures (PNG, 300 DPI)
- `output/tables/` -- all tables
- `output/references.bib` -- merged bibliography
- `PIPELINE_STATE.json` -- venue/style info

Read target venue from PIPELINE_STATE.json (`venue_style` field).

### 6.2 Prepare Inputs for Paper Sub-Skills

The `reference/sub-skills/paper-planning.md` procedure expects specific
root-level files (`NARRATIVE_REPORT.md`, `RESULTS_ANALYSIS.md`,
`references.bib`), NOT `output/manuscript.md`. Step 05 only guarantees
`output/manuscript.md`, `output/figures/`, `output/tables/`, and
`output/references.bib` exist — there are no separate `output/results.md`,
`output/methods.md`, or `output/literature_review.md` files.

Derive the sub-skill inputs directly from `output/manuscript.md` by
splitting it on its top-level Markdown headings:

```
# 1. Bibliography passes through unchanged
cp output/references.bib references.bib

# 2. RESULTS_ANALYSIS.md = the "## 5. Results" section of manuscript.md
#    (extract everything between "## 5. Results" and the next "## " heading)
Write RESULTS_ANALYSIS.md from output/manuscript.md (Results section only)

# 3. NARRATIVE_REPORT.md = a short summary derived from manuscript.md
Write NARRATIVE_REPORT.md containing:
  - Research question     (from PIPELINE_STATE.json)
  - Key findings          (3-5 bullets distilled from manuscript.md
                           "## 5. Results" section)
  - Models used           (1-2 sentences from manuscript.md
                           "## 4. Methods" section)
  - Literature context    (1-2 sentences from manuscript.md
                           "## 2. Related Work" section)

# 4. Ensure figures and tables are accessible at the project root
cp -r output/figures/ figures/
cp -r output/tables/ tables/
```

If any of the manuscript sections above is missing (e.g., the Related
Work heading was renamed), use whatever heading is closest in meaning
and note the substitution in `COMPILE_REPORT.md`. Do NOT block the
pipeline on a missing section — paper-planning can recover from a thin
narrative report.

### 6.3 Paper Planning

```
Read and execute reference/sub-skills/paper-planning.md with context: "$ARGUMENTS"
```

Output: `PAPER_PLAN.md`

### 6.4 Figure Conversion

```
Read and execute reference/sub-skills/figure-creating.md
```

Convert existing PNG figures to PDF vector format for LaTeX:
- For each `output/figures/*.png`:
  - If generated from Python code: re-render as PDF using the track's code
  - If re-rendering not feasible: convert PNG -> PDF at high quality
- Apply consistent sizing for the target venue
- Ensure colorblind-safe palettes preserved

Output: `paper/figures/*.pdf` + `paper/figures/*.png` (keep both formats)

### 6.5 LaTeX Writing

```
Read and execute reference/sub-skills/manuscript-writing.md with context: "$ARGUMENTS"
```

Convert `output/manuscript.md` sections into LaTeX:

| Markdown Section | LaTeX File | Notes |
|------------------|------------|-------|
| Abstract | `paper/sections/abstract.tex` | 150-250 words |
| 1. Introduction | `paper/sections/introduction.tex` | `\citet{}` / `\citep{}` citations |
| 2. Related Work | `paper/sections/related_work.tex` | Thematic organization |
| 3. Data & Setup | `paper/sections/data.tex` | Table 1 as `\begin{table}` |
| 4. Methods | `paper/sections/methods.tex` | Math notation via `\newcommand` |
| 5. Results | `paper/sections/results.tex` | `\begin{table}`, `\begin{figure}` |
| 6. Discussion | `paper/sections/discussion.tex` | Standard prose |
| References | `paper/references.bib` | Copy from `output/references.bib` |

**Venue-specific preamble**:

| Venue | Document Class | Key Packages |
|-------|----------------|--------------|
| NeurIPS | `\documentclass{article}` + neurips_2026.sty | neurips style |
| ICML | `\documentclass{article}` + icml2026.sty | icml style |
| ICLR | `\documentclass{article}` + iclr2026_conference.sty | iclr style |
| ACL | `\documentclass[11pt]{article}` + acl.sty | acl style |
| CVPR | `\documentclass[10pt,twocolumn]{article}` + cvpr.sty | cvpr style |
| AAAI | `\documentclass[letterpaper]{article}` + aaai26.sty | aaai style |
| General | `\documentclass[11pt]{article}` | amsmath, natbib, graphicx |

**LaTeX conventions**:
- Define `\newcommand` for model name, dataset names, metric names
- Tables: `\begin{table}[t]` with `\caption` above, `\label{tab:X}`
- Figures: `\begin{figure}[t]` with `\caption` below, `\label{fig:X}`
- Bold best results in tables
- Citations: `\citet{author2024}` for narrative, `\citep{author2024}` for parenthetical

**`paper/main.tex` structure**:
```latex
\documentclass[...]{...}
% Preamble (venue-specific)
\usepackage{amsmath,amssymb}
\usepackage{natbib}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
% Custom commands
\newcommand{\ours}{ModelName}
\newcommand{\method}[1]{\textsc{#1}}

\begin{document}
\title{...}
\author{[Author names -- left blank for user]}

\input{sections/abstract}
\input{sections/introduction}
\input{sections/related_work}
\input{sections/data}
\input{sections/methods}
\input{sections/results}
\input{sections/discussion}

\bibliographystyle{...}
\bibliography{references}

\appendix
\input{sections/appendix}
\end{document}
```

### 6.6 Compilation

```
Read and execute reference/sub-skills/paper-compiling.md
```

Compile LaTeX to PDF:
1. Pre-flight: verify all `\input` files exist, all figures referenced are present
2. Compile: `latexmk -pdf paper/main.tex`
3. If errors: diagnose, auto-fix (up to 3 attempts), recompile
4. Post-check: no `??` references, no missing figures, bibliography renders

Output: `paper/main.pdf` + `COMPILE_REPORT.md`

### 6.7 Update State

```json
{
  "stage": 6,
  "status": "completed",
  "latex_venue": "neurips",
  "pdf_pages": 10,
  "compile_status": "success",
  "compile_warnings": [],
  "timestamp": "..."
}
```

---

## Validation Checkpoints

| ID | Check Item | Pass Criteria | Failure Handling |
|----|------------|---------------|------------------|
| 6a | PAPER_PLAN.md exists | File created by paper-planning procedure | Regenerate from manuscript.md |
| 6b | PDF figures created | `paper/figures/*.pdf` for each figure | Fall back to PNG conversion |
| 6c | All .tex sections exist | One file per manuscript section | Regenerate missing section |
| 6d | main.tex compiles | PDF generated without errors | Auto-fix up to 3 times |
| 6e | No unresolved refs | No `??` in PDF | Fix `\ref` and `\cite` |
| 6f | Page count reasonable | Within venue limit (e.g., 9+refs for NeurIPS) | Note in log |
| 6g | Bibliography renders | All citations appear in references | Fix .bib entries |

---

## Next Step
-> Step 07: External Review via Codex MCP
