# Step 05: Manuscript Assembly

> **Executor**: Main Agent
> **Input**: All `output/` artifacts from Step 04 + `PIPELINE_STATE.json`
> **Output**: `output/manuscript.md`

---

## Execution Instructions

Read `reference/manuscript-template.md` for the exact Markdown structure.
Read `reference/assembly-rules.md` for stitching, numbering, and cross-referencing rules.

### 5.1 Pre-Assembly Inventory

Verify all required inputs exist:

| File | Source | Required |
|------|--------|----------|
| `output/methods.md` | Step 04 (merged) | Yes |
| `output/results.md` | Step 04 (merged) | Yes |
| `output/literature_review.md` | Step 04 Stream A | Yes |
| `output/analysis_strategy.md` | Step 03 | Yes |
| `output/tables/` | Step 04 (merged) | Yes |
| `output/figures/` | Step 04 (merged) | Yes |
| `output/references.bib` | Step 04 (merged) | Yes |
| `PIPELINE_STATE.json` | Steps 01-04 | Yes |

If any required file is missing, check `output/track_outputs/` for raw track outputs and attempt to recover.

### 5.2 Section Assembly

Assemble `output/manuscript.md` in this order. Write sections 1-6 first, then write the Abstract last.

#### Section 1: Introduction

Generate NEW content (do not copy from literature_review.md verbatim). Structure:

**Paragraph 1 -- Context & Significance**:
- What is the broad area? Why does it matter?
- Draw from `output/literature_review.md` background section

**Paragraph 2 -- What Is Known**:
- Summarize existing findings and approaches from literature review
- Cite key prior studies

**Paragraph 3 -- Gap & Motivation**:
- What remains unknown or under-explored?
- Draw from lit scan gaps + analysis strategy rationale
- Why do existing approaches fall short?

**Paragraph 4 -- This Study**:
- "In this work, we..." -- state the objective
- Briefly mention the modeling approach (informed by literature)
- State the hypothesis if applicable

**Paragraph 5 -- Contribution / Outline** (optional):
- What this study adds beyond prior work
- Brief roadmap of the paper

**Rules**:
- 3-5 paragraphs total, ~600-1000 words
- Cite 8-12 references (from literature_review.md)
- End with a clear statement of what THIS study does

#### Section 2: Related Work

Generate from literature_review.md, organized thematically:

**2.1 [Theme 1]**: e.g., "Pre-trained Language Models for [Task]"
- Summary of approaches, strengths, limitations

**2.2 [Theme 2]**: e.g., "Classical ML Approaches"
- How traditional methods compare

**2.3 [Theme 3]**: e.g., "[Domain]-Specific Methods"
- Domain-specific approaches and their limitations

**Rules**:
- Organize by theme, not chronologically
- Position our work relative to each theme
- End each subsection with how our approach differs

#### Section 3: Data & Experimental Setup

Generate from PIPELINE_STATE.json metadata:

**3.1 Data Description**:
- Dataset source, collection method, size
- Data modality and format
- Pre-processing pipeline

**3.2 Features / Inputs**:
- Input representation (tokenization, image transforms, feature engineering)
- Dimensionality, vocabulary size, resolution

**3.3 Experimental Protocol**:
- Train/val/test split ratios and strategy
- Evaluation metrics and their justification
- Hardware and compute budget
- Random seeds and reproducibility measures

**Generate Table 1** if not already in `output/tables/`:
```
Table 1: Dataset Statistics
| Split | N Samples | Class Distribution | Avg Length/Size |
|-------|-----------|-------------------|-----------------|
| Train | ... | ... | ... |
| Val | ... | ... | ... |
| Test | ... | ... | ... |
```

#### Section 4: Methods

Insert `output/methods.md` content.

Adjust ordering to match this standard flow:
1. Problem formulation
2. Baseline models (classical + standard DL)
3. Proposed / main models
4. Training details (optimizer, schedule, augmentation)
5. Ensemble / aggregation (if applicable)
6. Error analysis / interpretability approach

#### Section 5: Results

Insert `output/results.md` content.

Ensure results ordering matches methods ordering from Section 4.
Insert table and figure references at appropriate points:
- "Table N shows..." or "(Table N)"
- "Figure N displays..." or "(Figure N)"

Renumber all tables and figures sequentially starting from Table 1 / Figure 1.
Table 1 (dataset statistics) is in Section 3; results tables start from Table 2.

#### Section 6: Discussion

Generate NEW content. Read `reference/discussion-patterns.md` for structure.

**Paragraph 1 -- Key Findings Summary**:
- Restate main results in plain language
- Connect back to the research question

**Paragraph 2-3 -- Comparison with Prior Work**:
- Compare findings with key studies from literature review
- Where do our results agree/disagree?
- Why might differences exist?

**Paragraph 4 -- Methodological Strengths**:
- Multi-model approach (what each model family contributed)
- Cross-model convergence
- What our analytical strategy reveals beyond single-model studies

**Paragraph 5 -- Limitations**:
- Dataset size / domain coverage
- Compute constraints
- Generalizability
- Potential biases in data or evaluation

**Paragraph 6 -- Implications & Future Directions**:
- Practical implications of findings
- Recommendations for future research
- Promising directions identified by analysis

**Rules**:
- 5-7 paragraphs, ~800-1200 words
- Cite literature
- Do NOT overstate findings

#### Abstract (Write Last)

After all sections complete, write the abstract:

**Structure** (150-250 words):
- **Background**: 1-2 sentences (context + gap)
- **Objective**: 1 sentence (what this study does)
- **Methods**: 2-3 sentences (data, models, evaluation protocol)
- **Results**: 3-4 sentences (main findings with key numbers)
- **Conclusions**: 1-2 sentences (implications)

#### References

Append the full `output/references.bib` at the end.
Verify every citation in the text has an entry in references.
Verify no orphan references (in bib but not cited).

### 5.3 Final Manuscript Structure

```markdown
# [Title]

## Abstract
[150-250 words]

## 1. Introduction
[3-5 paragraphs, ~600-1000 words]

## 2. Related Work
### 2.1 [Theme 1]
### 2.2 [Theme 2]
### 2.3 [Theme 3]

## 3. Data and Experimental Setup
### 3.1 Data Description
### 3.2 Features / Inputs
### 3.3 Experimental Protocol

## 4. Methods
[From methods.md, reordered]

## 5. Results
[From results.md, with table/figure references]

## 6. Discussion
[5-7 paragraphs, ~800-1200 words]

## References
[From references.bib]

## Tables
[All tables, numbered sequentially]

## Figures
[All figure captions; actual PNGs in output/figures/]
```

### 5.4 Update State

```json
{
  "stage": 5,
  "status": "completed",
  "manuscript_word_count": 5000,
  "n_tables": 5,
  "n_figures": 8,
  "n_references": 25,
  "timestamp": "..."
}
```

---

## Validation Checkpoints

| ID | Check Item | Pass Criteria | Failure Handling |
|----|------------|---------------|------------------|
| 5a | All 6 sections present | Introduction through Discussion exist | Generate missing section |
| 5b | Abstract present | 150-250 words, no citations | Regenerate |
| 5c | Table numbering sequential | Table 1, 2, 3... no gaps | Renumber |
| 5d | Figure numbering sequential | Figure 1, 2, 3... no gaps | Renumber |
| 5e | All cited refs in bibliography | No [??] or missing citations | Add missing refs |
| 5f | No orphan references | Every bib entry cited somewhere | Remove orphans |
| 5g | Methods <-> Results alignment | Every model described has corresponding result | Flag gaps |
| 5h | manuscript.md written | File exists, >3000 words | Re-assemble |

---

## Next Step
-> Step 06: LaTeX Manuscript & PDF Compilation
