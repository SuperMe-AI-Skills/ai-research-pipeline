# Step 04: Parallel Execution -- Full Literature Review + Analysis Tracks

> **Executor**: Main Agent orchestrating parallel workers via `dispatch_track` (runtime-specific; see section 4.2.0 for the backend binding: `Agent` / `Task` / `spawn_agent` / sequential fallback).
> **Input**: `PIPELINE_STATE.json` + `output/analysis_strategy.md` + data file
> **Output**: `output/literature_review.md` + all track outputs in `output/track_outputs/`

---

## Execution Instructions

### 4.1 Preparation

Read from PIPELINE_STATE.json:
- `research_question`, `modality`, `discipline`, `venue_style`
- `variables` (target, inputs, metadata, subgroup)
- `testing_skill_path` (workflow steps 01-03)
- `analyzing_skill_path` (workflow steps 04-08)
- `method_tracks` (from Step 03)
- `data_file` path

**Important**: Workflow steps 01-03 live in the testing skill and steps 04-08
live in the analyzing skill. They are two separate skills for the same modality.
T1 (data diagnostics + baseline) reads from the testing skill; T2-T5 read from the
analyzing skill. See `reference/skill-routing-table.md`.

**REPO_ROOT discovery (portable, no hardcoded paths)**

Compute `REPO_ROOT` at runtime — never hardcode it. Try the following in order
and use the first one that resolves:

1. **Environment variable** — if `AIRESEARCH_ROOT` is set, use that.
2. **Relative to this orchestrator skill** — `REPO_ROOT` is the directory three
   levels above this very file:
   `<this file>/../../../` resolves to the AIResearch skill suite root.
   (You read this skill via `Read` so you already know its absolute
   path; strip the trailing `vera-ai-application-pipeline/workflow/step04-parallel.md`.)
3. **Sibling discovery** — search upward from `$PWD` for the first ancestor
   directory that contains BOTH `vera-ai-{modality}-testing/` AND
   `vera-ai-analysis-engine/`. That ancestor is `REPO_ROOT`.

Verify by checking that `{REPO_ROOT}/vera-ai-analysis-engine/` exists before proceeding.
If none of the methods resolves, abort with a clear error and ask the user
to set `AIRESEARCH_ROOT`.

```
REPO_ROOT      = <discovered as above>
TESTING_PATH   = {REPO_ROOT}/{testing_skill_path}            # e.g. vera-ai-nlp-testing
ANALYZING_PATH = {REPO_ROOT}/vera-ai-analysis-engine/{analyzing_skill_path}
                 # e.g. nlp/vera-ai-nlp-analyzing

Read both skills:
  {TESTING_PATH}/SKILL.md
  {TESTING_PATH}/workflow/01-collect-inputs.md
  {TESTING_PATH}/workflow/02-check-distribution.md
  {TESTING_PATH}/workflow/03-run-primary-test.md

  {ANALYZING_PATH}/SKILL.md
  {ANALYZING_PATH}/workflow/04-run-additional-models.md
  {ANALYZING_PATH}/workflow/05-analyze-subgroups.md
  {ANALYZING_PATH}/workflow/06-fit-advanced-models.md
  {ANALYZING_PATH}/workflow/07-compare-models.md
  {ANALYZING_PATH}/workflow/08-generate-manuscript.md
  {ANALYZING_PATH}/reference/
  {ANALYZING_PATH}/src/                  # reusable Python engine modules
```

Create output directories for each track listed in `method_tracks` from PIPELINE_STATE.json:
```
for each track in method_tracks:
  mkdir -p output/track_outputs/{track.id}/
```

Do NOT hardcode track IDs -- read them from state. The number and names of tracks
vary by modality (see reference/method-tracks.md).

### 4.2 Launch Streams Using Dependency Graph

#### 4.2.0 Resolve `dispatch_track` — the runtime-neutral parallel-worker operation

Before launching anything, detect the runtime's parallel-worker surface and
bind the abstract operation `dispatch_track(prompt)` to a concrete tool call.
Check in priority order (see `SKILL.md` → Tool Usage → "Parallel-worker
surface" for full rationale):

1. **`Agent`** (Claude Code / Claude Agent SDK):
   `dispatch_track(prompt)` → emit an `Agent` tool call. Launch ALL
   independent tracks by emitting multiple `Agent` calls in a single
   response; the runtime runs them in parallel.
2. **`Task`** (alternate Claude Code surface):
   `dispatch_track(prompt)` → emit a `Task` tool call. Same multi-call
   single-response pattern as `Agent`.
3. **`spawn_agent` + `send_input` + `wait_agent`** (lifecycle-based
   runtimes, e.g. Codex): `dispatch_track(prompt)` →
   `id = spawn_agent(); send_input(id, prompt)`. Collect results by
   calling `wait_agent(id)` for each spawned id in section 4.3.
4. **Sequential fallback** (NO parallel-worker tool available):
   `dispatch_track(prompt)` → run the prompt yourself as a subroutine of
   the main agent loop, one track at a time. Record
   `"dispatch_mode": "sequential"` in `PIPELINE_STATE.json` and emit a
   one-line note to `RESEARCH_LOG.md`. Wall-clock time increases but the
   output artifact set is identical.

Record the bound mode in `PIPELINE_STATE.json`:

```json
{"dispatch_mode": "Agent" | "Task" | "spawn_agent" | "sequential"}
```

The pipeline NEVER aborts because a specific parallel-worker tool is
missing — the sequential fallback is always available.

#### 4.2.1 Partition Tracks and Launch

Read `method_tracks` from PIPELINE_STATE.json. Partition into:
- **Independent tracks**: `depends_on` is null → launch immediately via
  `dispatch_track(prompt)` for each independent track.
- **Dependent tracks**: `depends_on` is non-null → launch ONLY after all
  dependencies complete (see section 4.3).

Launch ALL independent work using `dispatch_track` (not a specific tool
name). If `dispatch_mode` is `Agent` or `Task`, batch the calls into a
single response so the runtime can run them concurrently. If it is
`spawn_agent`, fan out the spawns and collect with `wait_agent` in 4.3.
If it is `sequential`, run them one after another.

#### Stream A: Full Literature Review

Dispatch one worker via `dispatch_track`:

```
Prompt: "Conduct a comprehensive literature review for the following research:

Research question: {research_question}
Discipline: {discipline}
Modality: {modality}
Models being used: {list from analysis_strategy.md}

Follow reference/sub-skills/literature-reviewing.md to search across arXiv, Google Scholar, Semantic Scholar, and conference proceedings.

Produce output/literature_review.md with:
1. Background & significance (2-3 paragraphs)
2. Prior modeling approaches (what architectures/methods others used, key findings)
3. Methodological justification (why our chosen models are appropriate)
4. Gaps this study addresses
5. Key references organized thematically

Also append any new references to output/references.bib.
Target: 15-25 well-chosen references."
```

#### Stream B: Analysis Tracks (one worker per independent track)

For each track marked `parallel: true` in the analysis strategy, invoke
`dispatch_track(prompt)` once. Each dispatched worker reads the relevant
workflow step files from the resolved source skill and executes them.

**Pre-populate inputs for all tracks** (so they don't ask interactively):
```
target_var: {from PIPELINE_STATE}
input_features: {from PIPELINE_STATE}
research_question: {from PIPELINE_STATE}
subgroup_var: {from PIPELINE_STATE}
data_file: {from PIPELINE_STATE}
modality: {from PIPELINE_STATE}
modality: {from PIPELINE_STATE}
discipline: {from PIPELINE_STATE}
venue_style: {from PIPELINE_STATE}
```

##### Dynamic Track Dispatch (for ALL independent tracks)

For EACH track in `method_tracks` where `depends_on` is null, construct
a worker prompt dynamically and launch it via `dispatch_track(prompt)`
using the backend bound in section 4.2.0. Do NOT use hardcoded track
names — read the track's ID, models, and workflow step mapping from
`output/analysis_strategy.md`.

**Routing rule** — choose the source skill based on which workflow steps the track uses:
- Track uses workflow steps **01, 02, or 03** → source = `TESTING_PATH` (testing skill)
- Track uses workflow steps **04, 05, 06, 07, or 08** → source = `ANALYZING_PATH` (analyzing skill)
- Reference files (`reference/specs/*`, `reference/rules/*`, `reference/patterns/*`) and
  the engine `src/` always come from `ANALYZING_PATH`.

For T1_baseline (data diagnostics + baseline model), the source is `TESTING_PATH`.
For T2-T5 (additional ML models, subgroups, deep learning, ensembles), the source is `ANALYZING_PATH`.

**Expanding `source_skill_path`**: the value written by Step 3 is a symbolic
constant (`TESTING_PATH` or `ANALYZING_PATH`). Step 4 must expand it to the
actual filesystem path resolved in section 4.1 BEFORE substituting into the
prompt template below:

```
resolved_source_path = {
    "TESTING_PATH":   TESTING_PATH,    # resolved from REPO_ROOT in 4.1
    "ANALYZING_PATH": ANALYZING_PATH,  # resolved from REPO_ROOT in 4.1
}[track.source_skill_path]
```

Then use `resolved_source_path` in the prompt below wherever
`{track.source_skill_path}` appears.

```
Prompt template for EACH independent track:

"Execute the '{track.id}' analysis track for:

[pre-populated inputs from above]

Track specification (from analysis strategy):
  Track ID: {track.id}
  Models/Methods: {track.models -- from analysis_strategy.md}
  Workflow steps: {track.workflow_steps -- from analysis_strategy.md}
  Source skill: {resolved_source_path -- expanded from track.source_skill_path}

Execute the workflow step file(s) listed above from {resolved_source_path}/workflow/.
For tracks whose source_skill_path is TESTING_PATH (T1_baseline), import models from
{ANALYZING_PATH}/src/ when the baseline needs the same engine code (e.g., evaluation
utilities, bootstrapped CIs).
Read {ANALYZING_PATH}/reference/ for reporting standards, sentence bank, and output variation protocol
regardless of which source skill the workflow steps came from.

Output to output/track_outputs/{track.id}/:
- methods.md ({track.id} methods description)
- results.md ({track.id} results with metrics, CIs, comparisons)
- code.py (training + evaluation code)
- figures/ (track-relevant plots, 300 DPI PNG)
- tables/ (if applicable)
- references.bib (methodological references for this track)

Follow reporting standards: report mean +/- std over seeds, always include baselines.
If this track involves neural models, report params and training time.
If N < 1000, note potential overfitting risk."
```

### 4.3 Wait for Independent Tracks and Launch Dependent Tracks

Collect each dispatched worker's output per the `dispatch_mode` bound in
section 4.2.0:
- `Agent` / `Task`: the multi-call response returns each worker's result
  in the tool-result block.
- `spawn_agent`: call `wait_agent(id)` for each id spawned in 4.2.1 and
  collect its output.
- `sequential`: each `dispatch_track` call has already returned by the
  time the next one starts.

As each worker completes:
1. Verify output files exist in the track's directory.
2. Log completion in `PIPELINE_STATE.json` (`tracks_completed` array).
3. Check for errors — if a track fails, log it in `RESEARCH_LOG.md` and
   continue with the remaining tracks.
4. **Check if any dependent track's prerequisites are now satisfied**:
   - For each pending track with `depends_on` non-null:
     - If ALL tracks in `depends_on` are now in `tracks_completed`,
       invoke `dispatch_track(prompt)` for that dependent track using
       the same bound mode from 4.2.0.

**For each dependent track**, construct its worker prompt by:
1. Reading the track's models from `output/analysis_strategy.md`.
2. Reading results from its dependency tracks:
   `output/track_outputs/{dep_id}/results.md`.
3. Reading the relevant workflow step from the correct source skill —
   apply the same routing rule (steps 01-03 from TESTING_PATH, steps
   04-08 from ANALYZING_PATH).

**Skip tracks** that were marked as not applicable in the analysis strategy.
Do NOT launch tracks whose dependencies failed — log the gap instead.

### 4.5 Convergence -- Model Comparison & Synthesis

After ALL tracks (from `method_tracks` in state) AND Stream A complete:

1. **Unified Model Comparison Table** (if >= 2 tracks produced metric results)
   Iterate over `tracks_completed` -- read metric results from each track's
   `output/track_outputs/{track_id}/results.md`.
   Build comparison table with all methods, metrics, params, training time.

2. **Cross-Method Insight Synthesis**
   For each completed track, extract its unique contribution from
   `output/track_outputs/{track_id}/results.md`.

   Write synthesis narrative (3-4 sentences):
   - What converges across models (if multiple independent tracks)
   - What each track uniquely reveals
   - Overall interpretation

3. **Merge Track Outputs**
   Iterate over `tracks_completed` in the order they appear in the analysis strategy.
   Combine outputs into unified files:
   - `output/methods.md` <- concatenate `{track_id}/methods.md` for each completed track
   - `output/results.md` <- concatenate `{track_id}/results.md` + synthesis narrative
   - `output/tables/` <- merge all track tables, renumber sequentially
   - `output/figures/` <- merge all track figures, renumber sequentially
   - `output/code.py` <- merge all track Python code with section headers per track
   - `output/references.bib` <- merge + deduplicate all track references

   For tracks that failed or were skipped: omit from merge, do not leave placeholders.

4. **Apply Output Variation Protocol**
   Read the analyzing skill's `reference/specs/output-variation-protocol.md` and `reference/specs/code-style-variation.md`
   (at `{ANALYZING_PATH}/reference/specs/`).
   Apply all six layers to the merged outputs.

### 4.6 Update State

Record the ACTUAL tracks that were completed/failed -- not a hardcoded list:

```json
{
  "stage": 4,
  "status": "completed",
  "tracks_completed": ["list of actual track IDs that completed"],
  "tracks_failed": ["list of actual track IDs that failed, if any"],
  "tracks_skipped": ["list of track IDs skipped as not applicable"],
  "lit_review_status": "completed",
  "lit_review_references": 22,
  "synthesis_complete": true,
  "timestamp": "..."
}
```

---

## Validation Checkpoints

| ID | Check Item | Pass Criteria | Failure Handling |
|----|------------|---------------|------------------|
| 4a | Stream A complete | `output/literature_review.md` exists | Proceed; note gap in manuscript |
| 4b | All independent tracks complete | Each completed track's directory has methods.md + results.md | Log failed tracks; continue |
| 4c | All dependent tracks complete | Each dependent track launched after deps satisfied | Log; skip in manuscript |
| 4d | Merged methods.md exists | `output/methods.md` non-empty, covers all completed tracks | Re-merge from track outputs |
| 4e | Merged results.md exists | `output/results.md` non-empty, covers all completed tracks | Re-merge from track outputs |
| 4f | Synthesis built (if applicable) | Model comparison table exists | Build from available tracks |
| 4g | Code files merged | `output/code.py` exists | Merge from track outputs |
| 4h | References merged | `output/references.bib` exists, no duplicates | Re-merge and dedup |

---

## Next Step
-> Step 05: Manuscript Assembly
