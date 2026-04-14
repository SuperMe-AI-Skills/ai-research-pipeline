# Step 03: Quick Literature Scan

> **Executor**: Main Agent
> **Input**: `PIPELINE_STATE.json` (research question, modality, discipline)
> **Output**: `output/lit_scan.md` + `output/analysis_strategy.md`

---

## Execution Instructions

### 3.1 Purpose

Before running our own analysis, understand how prior work has approached similar data and questions. This shapes which model tracks to include and ensures our analysis is grounded in established practice.

### 3.2 Fast Literature Survey

Read and execute `reference/sub-skills/literature-reviewing.md` with a focused query:

```
Read and execute reference/sub-skills/literature-reviewing.md with context: "{research_question} {modality} deep learning methods {discipline}"
```

Time-box this to 15 minutes maximum.

**Fallback if the lit scan fails (network unavailable, search returns
nothing useful, or the sub-skill errors out):** do NOT block the pipeline.
Skip directly to section 3.4 and copy the EXACT default track table for
the detected modality from `reference/method-tracks.md` with no
literature-informed adjustments. Set `lit_scan_status="skipped"` in the
state update (section 3.5) and add a one-line note to
`output/analysis_strategy.md` ("Lit scan unavailable; using default tracks
for {modality} from method-tracks.md") so downstream steps and the
manuscript know the literature context is thin. Do NOT invent citations.

Focus on:

1. **Analytical precedents**: What models have others used for this type of data in this domain?
   - Common architectures (e.g., "most NLP sentiment studies use BERT-based models")
   - Standard training approaches (e.g., "fine-tuning pre-trained models is standard for small datasets")
   - Emerging alternatives (e.g., "recent work uses prompt-based methods instead of fine-tuning")

2. **Reporting norms**: What do venues in this area expect?
   - Standard metrics (accuracy, F1, BLEU, mAP, etc.)
   - Baseline expectations
   - Ablation and analysis requirements

3. **Methodological gaps**: What analyses are common but rarely complemented?
   - e.g., "Most studies only fine-tune BERT; few compare with classical ML baselines"
   - This identifies opportunities for our analysis to add value

### 3.3 Produce Literature Scan

Write `output/lit_scan.md`:

```markdown
# Quick Literature Scan

## Research Question
[Verbatim from PIPELINE_STATE]

## How Others Have Analyzed Similar Data

### Common Models/Methods
- [Model 1]: Used in [cite 2-3 representative papers]
- [Model 2]: Used in [cite]
- [Model 3]: Emerging approach [cite]

### Reporting Conventions in [Domain]
- Standard metrics: [list]
- Expected baselines: [list]
- Analysis expectations: [ablation, error analysis, etc.]

### Gaps in Existing Analyses
- [Gap 1]: Most studies only use [X], missing [Y] perspective
- [Gap 2]: Few studies compare [DL model] with [classical baseline]

### Key References (for Introduction)
- [Author (Year)]: [1-sentence relevance]
- [Author (Year)]: [1-sentence relevance]
- ... (aim for 8-15 references)
```

### 3.4 Define Analysis Strategy

Based on the lit scan + modality, define which model tracks to run.

**CRITICAL**: Read `reference/method-tracks.md` and look up the EXACT track table for
this modality. Do NOT use a universal T1-T5 template. Track counts, names,
dependencies, and parallel/sequential status vary by modality:
- NLP may have T1 (classical), T2 (fine-tuned LM), T3 (few-shot/prompt), T4 (ensemble), T5 (error analysis)
- Structured may have T1 (classical ML), T2 (gradient boosting), T3 (neural net), T4 (ensemble), T5 (feature importance)
- Image may have T1 (CNN baseline), T2 (transfer learning), T3 (ViT), T4 (ensemble), T5 (interpretability)

Copy the exact track table from method-tracks.md for the detected modality,
then adjust based on literature scan:

- **Add tracks** if the literature shows a model is standard but not in defaults
- **Remove tracks** marked "skip" or "not applicable" for this modality
- **Adjust dependencies** if the literature suggests a different ordering
- **Note gaps** our analysis can fill

Write `output/analysis_strategy.md`:

```markdown
# Analysis Strategy

## Modality: [type]
## Informed By: Quick literature scan (Step 03)
## Source: reference/method-tracks.md -- [modality] section

## Model Tracks

[Copy the EXACT track table for this modality from method-tracks.md,
 then annotate with literature-informed adjustments. The `Workflow Steps`
 and `Source Skill Path` columns are MANDATORY — Step 04 reads them
 (as `track.workflow_steps` and `track.source_skill_path`) to dispatch
 the right SubAgent for each track. Use the canonical IDs from
 method-tracks.md verbatim (e.g. `T1_baseline`, `T2_ml`, `T3_deep`,
 `T4_ensemble`, `T5_subgroup` for NLP and structured; `T1_baseline`,
 `T2_transfer`, `T3_advanced`, `T4_ensemble`, `T5_subgroup` for image).
 Dependency cells must reference the SAME canonical IDs — not short
 forms like `T2, T3`.]

| Track | ID | Models/Methods | Workflow Steps | Source Skill Path | Depends On | Parallel? | Notes |
|-------|----|---------------|---------------|-------------------|------------|-----------|-------|
| [from method-tracks.md for this modality] | ... | ... | e.g. `01,02,03` | `TESTING_PATH` or `ANALYZING_PATH` | e.g. `T2_ml,T3_deep` | true/false | [lit-informed] |

**`Workflow Steps` column** (field name in state: `workflow_steps`):
comma-separated step numbers from `01` to `08`. T1 typically uses
`01,02,03`; T2/T3/T4 use steps from `04,05,06,07`; T5 uses `05`
(subgroup). Step 04 routes to TESTING_PATH for any track listing
01/02/03, otherwise ANALYZING_PATH (see step04-parallel.md routing rule).

**`Source Skill Path` column** (field name in state: `source_skill_path`):
literal symbolic constant `TESTING_PATH` if any workflow step in this row
is `01`/`02`/`03`, else `ANALYZING_PATH`. Step 04 reads this field by the
exact name `track.source_skill_path` and expands the constant to the
actual resolved path (`{REPO_ROOT}/{testing_skill_path}` or
`{REPO_ROOT}/vera-ai-analysis-engine/{analyzing_skill_path}`) at
dispatch time. Do NOT write literal filesystem paths here — always use
the symbolic constant.

## Dependency Graph
[Draw the ACTUAL dependency graph for this modality, using canonical IDs]

## Literature-Informed Adjustments
- [Adjustment 1]: Added/removed [model] because [lit reason]
- [Adjustment 2]: Changed dependency because [reason]

## Tracks Skipped
- [Track X]: Not applicable for [modality] -- [reason]
```

### 3.5 Update State

The `method_tracks` array must reflect the ACTUAL tracks for this modality,
not a universal template:

**Use the canonical track IDs from `reference/method-tracks.md` verbatim**.
Step 04 dispatches by exact-string match against these IDs and against the
`depends_on` entries — non-canonical or short-form IDs (e.g. `T1_classical`,
`T2`, `T3`) will silently strand dependent tracks. Example for NLP modality:

```json
{
  "stage": 3,
  "status": "completed",
  "method_tracks": [
    {"id": "T1_baseline",  "parallel": true,  "depends_on": null,
     "workflow_steps": ["01","02","03"], "source_skill_path": "TESTING_PATH"},
    {"id": "T2_ml",        "parallel": true,  "depends_on": null,
     "workflow_steps": ["04"],            "source_skill_path": "ANALYZING_PATH"},
    {"id": "T3_deep",      "parallel": true,  "depends_on": null,
     "workflow_steps": ["06"],            "source_skill_path": "ANALYZING_PATH"},
    {"id": "T4_ensemble",  "parallel": false, "depends_on": ["T2_ml","T3_deep"],
     "workflow_steps": ["06","07"],       "source_skill_path": "ANALYZING_PATH"},
    {"id": "T5_subgroup",  "parallel": false, "depends_on": ["T1_baseline"],
     "workflow_steps": ["05"],            "source_skill_path": "ANALYZING_PATH"}
  ],
  "lit_scan_references": 12,
  "timestamp": "..."
}
```

**Field name contract**: Step 04 reads `track.workflow_steps` and
`track.source_skill_path` by those exact names (see
`step04-parallel.md` section 4.2, prompt template at lines 153-156).
Any renaming here must be mirrored there.

For structured modality use the same IDs (`T1_baseline`, `T2_ml`, `T3_deep`,
`T4_ensemble`, `T5_subgroup`). For image modality the canonical IDs are
`T1_baseline`, `T2_transfer`, `T3_advanced`, `T4_ensemble`, `T5_subgroup`
— so the dependency lists must use those names (e.g. `T4_ensemble` depends
on `["T2_transfer","T3_advanced"]`).

---

## Validation Checkpoints

| ID | Check Item | Pass Criteria | Failure Handling |
|----|------------|---------------|------------------|
| 3a | Lit scan completed | `output/lit_scan.md` exists, non-empty | Proceed with default tracks; note limited lit context |
| 3b | Strategy defined | `output/analysis_strategy.md` exists with >= 2 tracks | Use default tracks from method-tracks.md |
| 3c | At least 1 launchable track | Tracks array has >= 1 entry with depends_on=null | Always true -- first track in any sequence has no deps |
| 3d | Dependencies valid | Sequential tracks reference existing parallel tracks | Fix dependency graph |

---

## Next Step
-> Step 04: Parallel Execution (Full Lit Review + Analysis Tracks)
