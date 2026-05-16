# Cross-Skill Interface

This repository uses paired reviewing and generating skills.

## AI Research Pairing

| Modality | Reviewing Skill | Generating Skill |
|---|---|---|
| nlp | `vera-ai-nlp-reviewing` | `vera-ai-nlp-generating` |
| structured | `vera-ai-structured-reviewing` | `vera-ai-structured-generating` |
| image | `vera-ai-image-reviewing` | `vera-ai-image-generating` |

## Handoff Contract

The reviewing skill owns input collection, diagnostics, baseline evidence, and
analysis readiness. The generating skill owns downstream model batteries,
subgroup analysis, model comparison, and manuscript-section artifact generation.

A pipeline handoff should preserve these fields in `PIPELINE_STATE.json` when
available:

- `modality`
- `testing_skill_path` or reviewing skill path
- `analyzing_skill_path` or generating skill path
- `data_file`
- `target` or target variable metadata
- `method_tracks`
- output directories and any baseline/diagnostic artifacts already produced

If a generating skill is invoked directly and these artifacts are missing, it
should stop and ask for the missing data/context instead of inventing upstream
results.
