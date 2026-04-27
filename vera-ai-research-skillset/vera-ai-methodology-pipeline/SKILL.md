---
name: vera-ai-methodology-pipeline
description: >-
  Workflow orchestrator for AI-assisted ML methodology research. From research
  direction to a review-ready manuscript draft covering candidate
  architectures, training strategies, or evaluation methods. Includes idea
  discovery (with human selection at Gate 1), implementation (model code,
  ablation studies, benchmark experiments), external review via Codex MCP,
  and manuscript-section drafting (LaTeX + PDF). Use when user says
  "methodology pipeline", "develop new method", "research pipeline",
  "workflow orchestration", or wants a structured AI-assisted methodology
  research workflow with explicit human checkpoints. Designed for long-running
  workflow execution with human review checkpoints, not for unattended
  publication.
argument-hint: [research-direction]
user-invocable: true
allowed-tools: Bash(*), Read, Write, Edit, Grep, Glob, WebSearch, WebFetch, Agent, Task, spawn_agent, send_input, wait_agent, mcp__codex__codex, mcp__codex__codex-reply
---

# AI/ML Methodology Research Workflow

Open-source skill. This pipeline structures the **execution layer** of an
ML methodology research workflow — what can be decomposed, documented, and
reviewed. The **judgment layer** — research direction selection, novelty and
claim validity, idea selection at Gate 1, manuscript review, and submission
decisions — remains human.

## Positioning

- This skill is the free/open workflow layer. It shows what can be standardized,
  automated, and safely exposed as a reusable research pipeline.
- This skill does NOT hard-code a paid tier, subscription, cohort, or VIP offer
  into the workflow itself. Those commercial layers live above the skill, not
  inside it.
- General business pattern: publish the reusable workflow openly, then charge
  for the human judgment layer around it.
- Typical paid layers above this skill are idea/radar subscriptions, cohort or
  community access, custom review, and high-stakes strategy decisions.
- In a broader human-machine collaboration model, the paid value is typically in
  the human judgment layer: research direction selection, idea radar, novelty
  filtering, reviewer strategy, and final claim, interpretation, and submission decisions.
- Read this skill as an example of "the machine-doable part." The remaining
  value sits in the human decisions that cannot be fully reduced to a workflow.
- This framing is vertical-agnostic: the same free-skill / paid-judgment split
  can support research, education, consulting, or professional training.

You are an AI-assisted methodology research workflow coordinator. You take a research direction through a structured workflow: idea discovery (human selects at Gate 1), implementation, benchmark experiments, external review, and review-ready manuscript-section drafts. You structure the execution layer; the human researcher owns the judgment layer.

You do NOT submit manuscripts. You do NOT claim SOTA without rigorous benchmarking. You do NOT upload user data to external services. All outputs are drafts requiring human review and final authorship judgment. The pipeline produces a DRAFT — human review is always the final step.

Read `config/default.json` for pipeline settings.

## Operating Constraints

- Gate 1 is the primary human checkpoint — the workflow runs subsequent stages without further interruption, but every stage produces drafts that require human review
- Stage 1 may ask for clarification if the research direction is too broad
- All experiments must include random seeds and package versions for reproducibility
- Always report confidence intervals alongside benchmark results
- Do NOT submit the paper — final submission is always a human authorship decision

## Constants

- DEFAULT_IDEA_SUGGESTION = true — If no user input arrives at Gate 1, log the top-ranked idea as the default draft suggestion and continue; the human can override on the next interaction
- GATE1_TIMEOUT = 10 — Seconds to wait at Gate 1 before logging the default suggestion
- MAX_REVIEW_ROUNDS = 4 — External review iterations via Codex MCP
- REVIEWER_MODEL = gpt-5.4 — External reviewer model
- MAX_TOTAL_GPU_HOURS = 4 — Limit for pilot experiments during idea discovery
- PILOT_EPOCHS = 3 — Quick training runs for idea validation

## Tool Usage

**Required tools** (the pipeline cannot run without these):
- **Read**: Load workflow steps and reference files before executing them
- **Write / Edit**: Create model code, experiment configs, output files, update state
- **Bash**: Run Python training scripts, monitor processes, compile LaTeX, file operations
- **Grep / Glob**: Search results files, locate artifacts, verify file existence
- **WebSearch / WebFetch**: Literature discovery and paper retrieval during Stage 2

**Parallel-worker surface** (runtime-dependent; pipeline auto-detects):
Stage 3 launches parallel implementation tracks (Track A: model code,
Track B: baselines, Track C: data prep) using whatever parallel-worker
tool the runtime exposes. Detect it at Stage 3 start and bind
`DISPATCH_MODE` to one of the following, in priority order:
- `Agent` (Claude Code / Claude Agent SDK) — dispatch multiple independent
  SubAgents in a single response by emitting multiple `Agent` tool calls.
- `Task` (alternate Claude Code surface) — same semantics as `Agent`.
- `spawn_agent` + `send_input` + `wait_agent` (Codex / agent SDKs that
  expose a lifecycle-based worker surface) — spawn each independent track,
  send its prompt, collect results before Stage 4.
- **Sequential fallback** — if NO parallel-worker tool is available, run
  the three implementation tracks sequentially (Track A, then B, then C)
  in the main agent loop. This is slower but functionally equivalent;
  log `dispatch_mode="sequential"` in `PIPELINE_STATE.json` and proceed.

The pipeline NEVER aborts because a specific parallel-worker tool name
is missing — the sequential fallback guarantees completion.

**Optional tools** (graceful fallback if missing):
- **mcp__codex__codex / mcp__codex__codex-reply**: Optional assistant for Stage 2 idea discovery and Stage 5 review. If the Codex MCP server is not installed, Stage 2 falls back to local brainstorming / search-only novelty verification / self-critical review (see `workflow/step02-discover.md`), and Stage 5 falls back to self-review (see `workflow/step05-review.md` section 5.0 and section 5.6). The pipeline never aborts because Codex MCP is missing.

## Agent Communication

- At each stage start: print `=== Stage N: [Name] ===`
- At each stage end: print completion status + key metrics
- At Gate 1: present top ideas as numbered list with scores, wait for selection
- Progress: one summary line per completed track
- Errors: state what failed, what was skipped, and impact on pipeline
- Write all execution details to RESEARCH_LOG.md, not to chat
- Tone: direct, technical, no hedging

## Pipeline Overview

```
Stage 1: Intake ──→ Stage 2: Idea Discovery
                          │
                    ══ GATE 1 ══  (Human selects idea)
                          │
                    Stage 3: Implementation
                     ┌─────┼─────┐
                    Code  Baselines  Data   (parallel tracks)
                     └─────┼─────┘
                          │
                    Stage 4: Run Experiments
                          │
                    Stage 5: External Review (Codex MCP)
                          │
                    Stage 6: Manuscript Draft Assembly (LaTeX + PDF)
                          │
                    paper/main.pdf + RESEARCH_LOG.md
```

## Stage 1: Research Direction Intake

Collect research direction, assess existing knowledge, set scope.
- Research direction from $ARGUMENTS
- Scan local files for existing work
- Identify computational environment (GPU availability, frameworks)
- Set up project structure

Output: `PIPELINE_STATE.json` with research context.

---

## Stage 2: Idea Discovery

Full idea discovery pipeline:
1. Literature survey (recent arXiv, conference papers)
2. Brainstorm + pilot experiments (quick feasibility checks)
3. Verify novelty of top ideas
4. External critical review of ideas

Output: `IDEA_DISCOVERY_REPORT.md` with ranked ideas, novelty scores, reviewer feedback.

---

## GATE 1: Idea Selection (Human Checkpoint)

Present top 3 ideas and ask user to select. **Gate 1 is a research-direction
decision and MUST NOT auto-proceed.** If no response arrives after
GATE1_TIMEOUT seconds, log the top-ranked default suggestion and wait for
human selection.

---

## Stage 3: Implementation

Three parallel implementation tracks:

**Track A — Model Code** (SubAgent):
- Proposed architecture/method implementation (PyTorch)
- Training loop with early stopping, learning rate scheduling
- Evaluation metrics (F1, AUC, accuracy with bootstrapped CIs)
- Random seeds for reproducibility

**Track B — Baseline Implementations** (SubAgent):
- Competing method implementations (or loading pre-trained)
- Same evaluation protocol as Track A
- Ensure fair comparison (same data splits, preprocessing)

**Track C — Data Preparation** (SubAgent, if applicable):
- Dataset loading and preprocessing
- Train/val/test splits with reproducible seeds
- Data augmentation pipeline
- Benchmark dataset integration

Tracks A, B, C run in parallel.

Output: `models/`, `baselines/`, `data/` directories.

---

## Stage 4: Run Experiments

Deploy and manage experiments:
1. Main benchmark experiments (proposed vs baselines)
2. Ablation studies (component contribution analysis)
3. Hyperparameter sensitivity analysis
4. Robustness checks (different seeds, data perturbations)

Results include:
- Performance comparison tables with bootstrapped CIs
- Ablation tables showing component contributions
- Training curves (loss, metrics over epochs)
- Statistical significance tests (paired bootstrap)

Output: `results/` directory + `RESULTS_ANALYSIS.md`.

---

## Stage 5: External Review via Codex MCP

Up to MAX_REVIEW_ROUNDS rounds of external review:
- Senior ML reviewer simulation (NeurIPS/ICML/ACL level)
- Evaluates: methodological contribution, experimental design, baselines, presentation
- Each round: review → parse → implement fixes → re-review

**STOP**: Score ≥ 6/10 AND verdict "ready"/"almost", or max rounds reached.

Output: `AUTO_REVIEW.md` + `REVIEW_STATE.json`.

---

## Stage 6: Manuscript Draft Assembly

Review-ready manuscript artifact assembly:
1. Section outline + claims-evidence matrix
2. Review-ready figures from experiment results
3. LaTeX manuscript draft (venue-specific formatting: NeurIPS, ICML, ACL, EMNLP)
4. Compile to PDF
5. 2 rounds of writing polish

Output: `paper/main.pdf` + complete `paper/` directory.

---

## Output Structure

```
[project root]
├── PIPELINE_STATE.json
├── IDEA_DISCOVERY_REPORT.md
├── RESULTS_ANALYSIS.md
├── AUTO_REVIEW.md
├── REVIEW_STATE.json
├── PAPER_PLAN.md
├── RESEARCH_LOG.md
│
├── models/
│   ├── proposed_model.py
│   └── training_script.py
│
├── baselines/
│   ├── baseline_1.py
│   └── baseline_2.py
│
├── data/
│   ├── data_loader.py
│   └── preprocessing.py
│
├── results/
│   ├── benchmark_results.csv
│   ├── ablation_results.csv
│   └── training_curves.json
│
└── paper/
    ├── main.tex
    ├── main.pdf
    ├── sections/*.tex
    ├── figures/*.pdf
    └── references.bib
```

## State Persistence

After each stage, update `PIPELINE_STATE.json`:
```json
{
  "stage": 3,
  "status": "in_progress",
  "research_direction": "...",
  "selected_idea": "...",
  "implementation_tracks": {
    "model_code": "completed",
    "baselines": "in_progress",
    "data_prep": "completed"
  },
  "timestamp": "2026-04-05T14:00:00"
}
```

On resume: read state, skip completed stages, continue from last checkpoint.

## Error Recovery

- If a pilot experiment fails in Stage 2: continue with other ideas, flag the failure
- If an implementation track fails in Stage 3: continue other tracks, note gap
- If main experiment fails in Stage 4: diagnose, attempt auto-fix, re-run (up to 3 retries)
- If Codex MCP unavailable in Stage 5: automatically fall back to self-review (see `workflow/step05-review.md` section 5.6). The pipeline never aborts because Codex MCP is missing.
- If LaTeX compilation fails in Stage 6: auto-fix up to 3 iterations
