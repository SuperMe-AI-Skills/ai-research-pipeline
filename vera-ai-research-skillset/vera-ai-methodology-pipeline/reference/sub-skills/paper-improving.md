# Paper Improving: Writing Polish and Quality Enhancement

2 rounds of writing-focused improvement for the compiled manuscript.

## Input

- `paper/main.tex` and `paper/sections/*.tex` -- compiled manuscript
- `paper/main.pdf` -- current PDF
- `PAPER_PLAN.md` -- claims-evidence matrix

## Improvement Rounds

### Round 1: Structural and Content Review

| Check | What to Look For | Fix |
|-------|-----------------|-----|
| Claims-evidence alignment | Every claim in intro has matching evidence | Add evidence or weaken claim |
| Abstract-conclusion sync | Abstract claims match conclusion | Synchronize |
| Notation consistency | Same symbol means same thing throughout | Standardize via `\newcommand` |
| Table formatting | Aligned decimals, bold best, +/- spacing | Fix formatting |
| Figure quality | Labels readable, colors distinguishable | Regenerate if needed |
| Citation accuracy | Correct years, author names, venues | Verify against source |
| Cross-references | No ?? in text, all labels resolve | Fix `\ref` and `\cite` |
| Page budget | Within venue limits | Trim or move to appendix |

### Round 2: Prose and Style Polish

| Check | What to Look For | Fix |
|-------|-----------------|-----|
| AI writing artifacts | "delve", "leverage", "harness", "utilize", "facilitate" | Replace with natural alternatives |
| Passive voice excess | "It was observed that..." | Convert to active: "We observed..." |
| Vague quantifiers | "significant improvement" without numbers | Add specific metrics |
| Overclaiming | "state-of-the-art" without evidence | Soften or provide evidence |
| Redundancy | Same point made in multiple sections | Consolidate |
| Transitions | Abrupt section changes | Add connecting sentences |
| Tense consistency | Mixed present/past in same section | Standardize |
| Jargon overload | Too many undefined terms | Define on first use or simplify |
| Paragraph structure | Paragraphs > 8 lines | Break into focused paragraphs |
| Sentence length | Sentences > 40 words | Split for clarity |

### De-AI Checklist

Replace these AI-typical patterns:

| AI Pattern | Replace With |
|-----------|-------------|
| "It is worth noting that" | Remove (just state the point) |
| "In this regard" | Remove or use "Here" |
| "A plethora of" | "Many" or "Several" |
| "Leverage" | "Use" |
| "Utilize" | "Use" |
| "Facilitate" | "Enable" or "Support" |
| "Delve into" | "Examine" or "Study" |
| "Paradigm" | "Approach" or "Framework" |
| "Harness" | "Use" or "Apply" |
| "Cutting-edge" | "Recent" or name the specific advance |
| "Paves the way" | "Enables" or describe specifically |
| "Sheds light on" | "Clarifies" or "Reveals" |

## Scoring

After each round, score the manuscript 1-10:

| Score | Meaning |
|-------|---------|
| 9-10 | Ready for submission |
| 7-8 | Minor issues, mostly cosmetic |
| 5-6 | Needs content fixes (missing evidence, unclear claims) |
| 3-4 | Major structural issues |
| 1-2 | Needs substantial rewriting |

## Output

- Updated `paper/sections/*.tex` files
- Recompiled `paper/main.pdf`
- Score progression: Round 0 -> Round 1 -> Round 2

## Key Rules

- Do NOT change experimental results or methodology
- Do NOT add new claims not supported by existing evidence
- Focus on clarity, conciseness, and accuracy
- Every change should make the paper clearer, not longer
- Flag any [VERIFY] or [TODO] markers that remain
