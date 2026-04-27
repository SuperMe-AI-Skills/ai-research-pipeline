# Modality Detection Rules

Three-signal system for auto-detecting the data modality (NLP, Structured, Image) from data characteristics, file metadata, and research question context.

This skill suite supports only three routed primary modalities:
- `nlp`
- `structured`
- `image`

If the evidence instead points to audio, graph, or true multimodal data, mark
the task as unsupported for this pipeline and stop before routing.

## Signal 1: Primary — Data Characteristics

Inspect the input data's structure and content.

### Decision Tree

```
input data
│
├── Is it a directory of image files? (PNG, JPEG, TIFF, DICOM)
│   └── YES → image
│
├── Is it a directory with subdirectories per class? (ImageFolder format)
│   └── YES → image
│
├── Does it contain a column with free-text (avg length > 20 words)?
│   └── YES → nlp
│
├── Is it a CSV/DataFrame with primarily numeric/categorical columns?
│   └── YES → structured
│
├── Does the task mention image files, pixels, or visual data?
│   └── YES → image
│
├── Does the task mention text, documents, reviews, or language?
│   └── YES → nlp
│
└── Ambiguous → flag for user confirmation
```

### NLP Detection Sub-Rules

| Property | Test | Threshold | Result |
|----------|------|-----------|--------|
| Mean text length | `mean(word_count(text_col))` | > 20 words | nlp (likely) |
| Text column count | columns with dtype=object and long strings | ≥ 1 | nlp (possible) |
| Vocabulary diversity | unique tokens / total tokens | > 0.05 | nlp (supports) |

### Structured Detection Sub-Rules

| Property | Test | Threshold | Result |
|----------|------|-----------|--------|
| Numeric columns | count of numeric dtype columns | > 50% of total | structured (likely) |
| Categorical columns | count of categorical columns with < 50 levels | > 0 | structured (supports) |
| No text columns | all columns are numeric/categorical with short values | — | structured |

### Image Detection Sub-Rules

| Property | Test | Threshold | Result |
|----------|------|-----------|--------|
| File extensions | files end with .png, .jpg, .jpeg, .tiff, .dicom | > 90% | image |
| Directory structure | subdirectories contain image files | — | image (ImageFolder) |
| Pixel arrays | data contains 3D/4D arrays (H x W x C) | — | image |

---

## Signal 2: Secondary — File & Column Name Heuristics

Scan file names, column names, and metadata for keywords.

| Keyword Pattern | Inferred Modality |
|-----------------|-------------------|
| "text", "review", "comment", "document", "abstract", "content", "tweet", "post" | nlp |
| "sentence", "paragraph", "corpus", "vocabulary", "token" | nlp |
| "image", "img", "photo", "frame", "scan", "xray", "mri", "ct_scan" | image |
| "pixel", "resolution", "rgb", "channel", "thumbnail" | image |
| "feature_1", "feature_2", "age", "income", "score", "count", "amount" | structured |
| "category", "type", "class", "label", "target", "diagnosis" | structured (ambiguous) |

---

## Signal 3: Contextual — Research Question Text

Parse the research question for modality-specific keywords.

| Research Question Pattern | Inferred Modality |
|---------------------------|-------------------|
| "classify text", "sentiment analysis", "NLP", "language model", "AI detection" | nlp |
| "document classification", "spam detection", "topic modeling", "text mining" | nlp |
| "image classification", "object recognition", "medical imaging", "CNN" | image |
| "computer vision", "visual recognition", "X-ray", "pathology" | image |
| "predict from features", "tabular data", "structured data", "CSV" | structured |
| "feature importance", "regression", "customer churn", "credit scoring" | structured |

---

## Confidence Scoring

| Level | Condition | Action |
|-------|-----------|--------|
| **HIGH** | All 3 signals agree, OR Signal 1 is unambiguous | Log default routing suggestion after timeout; user may correct before final interpretation |
| **MEDIUM** | 2 of 3 signals agree | Present detection, ask to confirm |
| **LOW** | Signals conflict or ambiguous | Present candidates, require user selection |

### Mixed-Modality Resolution

| Ambiguity | Resolution |
|-----------|------------|
| Text + tabular columns | If primary task is text classification → nlp (extra features available). If primary task is prediction from columns → structured. |
| Image + metadata | If primary task is image classification → image. Metadata can be extra features. |
| Text + images | Ask user to choose a supported primary modality; otherwise stop as unsupported multimodal input. |

---

## Routing Table

| Detected Modality | Testing Skill | Analyzing Skill |
|-------------------|---------------|-----------------|
| nlp | vera-ai-nlp-testing | vera-ai-nlp-analyzing |
| structured | vera-ai-structured-testing | vera-ai-structured-analyzing |
| image | vera-ai-image-testing | vera-ai-image-analyzing |
