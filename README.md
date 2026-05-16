# AI Research Pipeline

Open-source AI research skills for Claude Code and Codex-compatible workflows.

This repository packages the Vera AI research skill family for data diagnostics, model generation, interpretability checks, manuscript-section drafting, LaTeX-ready artifact assembly, and review checkpoints across text, structured/tabular, and image research projects.

Vera structures the execution layer. Researchers own the question, scientific interpretation, and submission decisions.

## Skill Inventory

### Reviewing Skills

| Skill | Modality | Role |
|---|---|---|
| `vera-ai-image-reviewing` | Image | Review inputs, diagnostics, baseline evidence, reporting constraints, and analysis readiness. |
| `vera-ai-nlp-reviewing` | NLP | Review inputs, diagnostics, baseline evidence, reporting constraints, and analysis readiness. |
| `vera-ai-structured-reviewing` | Structured | Review inputs, diagnostics, baseline evidence, reporting constraints, and analysis readiness. |

### Generating Skills

| Skill | Modality | Role |
|---|---|---|
| `vera-ai-image-generating` | Image | Run candidate model workflows and generate methods/results artifacts with traceable outputs. |
| `vera-ai-nlp-generating` | NLP | Run candidate model workflows and generate methods/results artifacts with traceable outputs. |
| `vera-ai-structured-generating` | Structured | Run candidate model workflows and generate methods/results artifacts with traceable outputs. |

### Pipeline Skills

| Skill | Workflow | Role |
|---|---|---|
| `vera-ai-application-pipelining` | Application | Coordinate multi-step research execution, artifact assembly, and review checkpoints. |
| `vera-ai-methodology-pipelining` | Methodology | Coordinate multi-step research execution, artifact assembly, and review checkpoints. |

## Install

Clone the repository:

```bash
git clone https://github.com/VeraSuperHub/ai-research-pipeline.git
```

Claude Code users can import `vera-ai-research.plugin` with the Claude Code plugin flow.

Codex users can install the extracted skill folders directly:

```bash
mkdir -p ~/.codex/skills
cp -R vera-ai-research-skillset/vera-ai-* ~/.codex/skills/
```

You can also copy a single folder from `vera-ai-research-skillset/` if you only need one skill.

## Repository Contents

| Path | Purpose |
|---|---|
| `vera-ai-research-skillset/` | Extracted skill folders for direct inspection, editing, or Codex installation. |
| `vera-ai-research.plugin` | Claude Code plugin bundle rebuilt from the same extracted skills. |
| `PLATFORM-COMPATIBILITY.md` | Runtime mapping for Claude Code, Codex, and fallback behavior. |
| `CROSS-SKILL-INTERFACE.md` | Handoff contract between reviewing and generating skills. |
| `requirements.txt` | Shared Python dependencies used by the skill scripts. |

## Workflow Shape

```text
Reviewing skills -> Generating skills -> Pipeline skills
      |                    |                   |
 diagnostics       model/artifact runs   manuscript assembly
 readiness checks  comparisons           review checkpoints
```

The skills are designed to support reproducible execution, not to replace expert judgment. Treat generated methods, results, and interpretation drafts as reviewable artifacts that need domain-owner approval before use.

## License

GPL-3.0. See [LICENSE](LICENSE).
