# Assembly Rules

Rules for stitching track outputs into a unified manuscript.

## Table Numbering

1. Table 1 is ALWAYS "Dataset Description" (generated in Section 2)
2. Subsequent tables numbered sequentially in order of first mention in text
3. Track-internal numbering replaced with manuscript-global numbering

**Renumbering procedure**:
1. Inventory all tables from `output/tables/` and `output/track_outputs/*/tables/`
2. Assign global numbers based on manuscript mention order
3. Update all in-text references
4. Rename files: `table_01_dataset.md`, `table_02_baseline.md`, etc.

## Figure Numbering

1. Figures numbered sequentially in order of first mention
2. Follow same section order as tables
3. Track-internal numbering replaced with global numbering

**Renumbering procedure**:
1. Inventory all figures from `output/figures/` and `output/track_outputs/*/figures/`
2. Assign global numbers based on manuscript mention order
3. Update all in-text references
4. Rename files: `figure_01_class_balance.png`, `figure_02_baseline_roc.png`, etc.

## Methods Section Ordering

Merge track methods fragments in this standard order:

1. **Data description and preprocessing** (from T1)
2. **Baseline model** (from T1)
3. **ML model battery** (from T2)
4. **Deep learning models** (from T3)
5. **Ensemble / advanced methods** (from T4)
6. **Subgroup analysis** (from T5)
7. **Model comparison approach** (from convergence step)
8. **Evaluation protocol** (metrics, bootstrapping, train/val/test)
9. **Software** (merged from all tracks, deduplicated)

If a track was skipped, omit its section.

## Results Section Ordering

Mirror the methods ordering exactly:

1. **Dataset summary** (Table 1 reference, key statistics)
2. **Baseline results** (from T1)
3. **ML model results** (from T2)
4. **Deep learning results** (from T3)
5. **Ensemble results** (from T4)
6. **Subgroup results** (from T5)
7. **Cross-method comparison** (unified table + narrative synthesis)

## Code Merging

Merge track code files into unified `output/code.py`:

```python
# ============================================================
# AI/ML Analysis Code
# Research Question: {research_question}
# Data Modality: {modality}
# ============================================================

# -- Section 1: Setup & Data Loading -------------------------
# [from T1 setup code]

# -- Section 2: Data Diagnostics & Baseline -------------------
# [from T1 analysis code]

# -- Section 3: ML Model Battery ------------------------------
# [from T2 code]

# -- Section 4: Deep Learning Models --------------------------
# [from T3 code]

# -- Section 5: Ensemble / Advanced ---------------------------
# [from T4 code]

# -- Section 6: Subgroup Analysis -----------------------------
# [from T5 code]

# -- Section 7: Model Comparison ------------------------------
# [from convergence code]
```

**Rules**:
- Data loading appears ONCE at the top (deduplicate)
- Import statements consolidated at the top (deduplicate)
- Random seed set once at the beginning
- Each section clearly labeled with separator comments
- Apply style variation AFTER merging

## Reference Merging

1. Collect all `references.bib` files from tracks + literature review
2. Deduplicate by DOI (preferred) or by author+year+title match
3. Verify every citation in manuscript text has a bib entry
4. Remove orphan entries (in bib but not cited)
5. Sort alphabetically by first author surname

## Cross-Track Consistency Checks

After merging, verify:

| Check | How |
|-------|-----|
| Model names consistent | Same model named identically across all sections |
| N consistent | Same dataset size reported everywhere |
| Metric directions agree | If T2 best model has high F1, T3 comparison should reflect this |
| Method names match | "logistic regression" in Methods = "logistic regression" in Results |
| Abbreviations defined | First use in manuscript is spelled out |
| Feature names consistent | Same features referenced across all tracks |

## Handling Missing Tracks

If a track failed or was skipped:

1. Omit the corresponding Methods and Results subsections
2. Do NOT leave placeholder text
3. Adjust subsection numbering to remain sequential
4. Note the omission in RESEARCH_LOG.md
5. In Discussion, do not reference methods that weren't run

## Output Variation Protocol Application

After all merging is complete, apply the analyzing skill's output variation protocol:

1. Read `reference/specs/output-variation-protocol.md` from the analyzing skill
2. Apply sentence bank variations to merged prose (Layer 1)
3. Randomize table/figure naming conventions (Layer 2)
4. Vary interpretation depth per section (Layer 3)
5. Apply code style variation to merged code file (Layer 4)
6. Ensure methods section doesn't expose pipeline logic (Layer 5)
7. The multi-track orchestration itself is Layer 6

Apply style variation LAST — after all code is merged and finalized.
