# Step 02: Modality Detection & Routing

> **Executor**: Main Agent
> **Input**: `PIPELINE_STATE.json` from Step 01
> **Output**: Confirmed modality type + analysis strategy baseline

---

## Execution Instructions

### 2.1 Three-Signal Detection

Read `reference/modality-detection-rules.md` for the full decision tree.

Apply three signal layers to the input data:

**Scope guard**: this skill suite routes only three supported primary
modalities: `nlp`, `structured`, and `image`. If signals suggest `audio`,
`multimodal`, or `graph`, treat that as **unsupported for this pipeline** and
stop before routing.

#### Signal 1: Primary (Data Characteristics)

| Data Property | Inferred Modality |
|---------------|-------------------|
| Text columns present, avg token length > 10 | nlp |
| Only numeric/categorical columns, no text/image/audio | structured |
| Image paths or image directory structure | image |
| Audio paths or audio files | unsupported (audio) |
| Text + numeric/categorical mixed inputs | nlp (with structured features) |
| Image + text paired inputs | unsupported (multimodal) |
| Graph/network data structure | unsupported (graph) |
| Sequential numeric data (time series) | structured (timeseries variant) |

#### Signal 2: Secondary (Column Name & Content Heuristics)

Scan column names and content for domain signals:
- "text", "review", "comment", "description", "sentence", "document" -> nlp
- "image", "img", "photo", "path", "url" (pointing to images) -> image
- "audio", "wav", "mp3", "speech", "waveform" -> unsupported (audio)
- Column contains long strings (avg > 50 chars) -> nlp
- Column contains file paths ending in .jpg/.png/.bmp -> image
- All columns are short numeric/categorical -> structured

#### Signal 3: Contextual (Research Question Text)

Parse the research question for analytical intent:
- "text classification", "sentiment", "NER", "translation", "summarization" -> nlp
- "image classification", "object detection", "segmentation" -> image
- "predict from features", "regression on tabular data", "feature importance" -> structured
- "speech recognition", "speaker identification" -> unsupported (audio)
- "visual question answering", "image captioning" -> unsupported (multimodal)

### 2.2 Confidence Scoring

| Confidence | Condition | Action |
|------------|-----------|--------|
| HIGH | All 3 signals agree on same modality | Present detection; log default routing suggestion after DEFAULT_SUGGESTION_TIMEOUT and continue; user may correct before final interpretation |
| MEDIUM | 2 of 3 signals agree | Present detection with alternatives, ask user to confirm |
| LOW | All 3 signals disagree, or signals ambiguous | Present top candidates, require user selection |

### 2.3 Human Gate

If the top candidate is unsupported (`audio`, `multimodal`, or `graph`), do
NOT continue to routing. Instead present:

```text
Detected primary modality: [unsupported type]

This AIResearch suite currently supports only:
1. NLP text classification
2. Structured/tabular prediction
3. Image classification

Please either:
1. Choose the supported primary modality to analyze
2. Narrow the task to one supported modality
3. Stop here and use a different workflow
```

Only supported modalities may proceed to the confirmation template below.

Present to user:

```
Modality detected: [NLP / Structured / Image] (confidence: [HIGH/MEDIUM/LOW])

Signals:
  Data characteristics -> [modality]
  Column names/content -> [modality]
  Research question -> [modality]

This means I'll use: [describe the analysis approach briefly]

Confirm? (logging default routing suggestion in Xs if HIGH confidence; you can correct before final interpretation)
  1. Yes, proceed with [MODALITY]
  2. Actually, it's [alternative modality]
  3. Let me explain...
```

If HIGH confidence and no user response within DEFAULT_SUGGESTION_TIMEOUT: log the default routing suggestion and continue.
If user corrects (now or before final interpretation): update modality and re-route.

### 2.4 Route to Analysis Skill

Look up the confirmed modality in `reference/skill-routing-table.md`.
Record the skill path in PIPELINE_STATE.json.

### 2.5 Update State

```json
{
  "stage": 2,
  "status": "completed",
  "modality": "nlp",
  "detection_confidence": "HIGH",
  "detection_signals": {
    "primary": "nlp",
    "secondary": "nlp",
    "contextual": "nlp"
  },
  "testing_skill_path": "vera-ai-nlp-testing",
  "analyzing_skill_path": "nlp/vera-ai-nlp-analyzing",
  "timestamp": "..."
}
```

The two paths resolve to:
- `{testing_skill_path}/`              → workflow steps 01-03
- `vera-ai-analysis-engine/{analyzing_skill_path}/`  → workflow steps 04-08 + src/

Per-modality routing (from `reference/skill-routing-table.md`):

| Modality | testing_skill_path | analyzing_skill_path |
|----------|--------------------|----------------------|
| nlp | `vera-ai-nlp-testing` | `nlp/vera-ai-nlp-analyzing` |
| structured | `vera-ai-structured-testing` | `structured/vera-ai-structured-analyzing` |
| image | `vera-ai-image-testing` | `image/vera-ai-image-analyzing` |

---

## Validation Checkpoints

| ID | Check Item | Pass Criteria | Failure Handling |
|----|------------|---------------|------------------|
| 2a | Supported modality determined | One of `nlp`, `structured`, `image` | Ask user to narrow or stop as unsupported |
| 2b | User confirmed (or default routing logged) | Confirmation received or timeout elapsed | Wait for user |
| 2c | Skill path valid | SKILL.md exists at resolved path | Check routing table; ask user |
| 2d | State updated | PIPELINE_STATE.json has stage=2 fields | Rewrite |

---

## Next Step
-> Step 03: Quick Literature Scan
