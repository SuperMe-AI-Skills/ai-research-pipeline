# Discussion Section Patterns

Patterns for generating the Discussion section. The discussion is always generated NEW (not from analysis skill output) because it requires synthesis across all tracks and literature.

## Structure: 6-Paragraph Pattern

### Paragraph 1: Key Findings Summary

**Purpose**: Restate main results in plain language without raw metrics.

**Pattern**:
```
Our analysis of [dataset description] using [N] modeling approaches revealed that
[main finding 1]. [Main finding 2, if applicable]. These findings were consistent
across [multiple model families / classical ML, neural, and transformer-based methods],
strengthening confidence in the observed [patterns/predictions/associations].
```

**Rules**:
- No raw metric values in this paragraph (save for Results)
- Use "achieves", "outperforms", "is comparable to" language
- Reference the research question explicitly
- Mention cross-model convergence if applicable

### Paragraphs 2-3: Comparison with Prior Work

**Purpose**: Situate findings within existing literature.

**Pattern for agreement**:
```
Our finding that [model X outperforms baseline Y] is consistent with [Author (Year)],
who reported [similar finding] on [their dataset]. Similarly, [Author (Year)]
found [related result] using [their approach], suggesting [broader pattern].
```

**Pattern for disagreement**:
```
In contrast to [Author (Year)], who found [different result], our analysis suggests
[our finding]. This discrepancy may reflect [difference in dataset / domain /
scale / evaluation protocol / model configuration].
```

**Pattern for extension**:
```
While [Author (Year)] demonstrated [their finding] using [their approach], our
multi-model analysis extends this by revealing [new insight from ablation/comparison].
Specifically, [our analysis] showed [pattern not visible in single-model studies],
suggesting [interpretation].
```

**Rules**:
- Compare with at least 3 prior studies from literature_review.md
- Always explain WHY results might differ
- Highlight what our multi-model approach adds beyond prior single-model studies

### Paragraph 4: Methodological Strengths

**Purpose**: Articulate the value of the analytical approach.

**Pattern**:
```
A strength of this analysis is the multi-model approach. [Classical ML baseline]
provided [specific insight -- interpretable feature importance, decision boundaries].
[Neural network approach] captured [nonlinear patterns / representation quality].
[Transformer/pre-trained model] leveraged [pre-training knowledge / contextual
representations]. The convergence of [key finding] across model families
strengthens confidence that the observed [performance/pattern] is robust to
architectural choices.
```

**Rules**:
- Name each model family and its unique contribution
- Reference the model comparison table if applicable
- Use "complement" and "converge" language -- models are lenses, not competitors
- If a model disagreed with others, frame it as revealing complexity

### Paragraph 5: Limitations

**Purpose**: Honest accounting of study weaknesses.

**Minimum 3 limitations**. Select from these categories:

| Category | Example Limitation |
|----------|-------------------|
| Data size | "The dataset of N={X} samples may limit the capacity to train deep models; results may improve with larger datasets" |
| Domain specificity | "These findings are specific to [domain]; generalization to [other domain] requires further validation" |
| Compute | "Compute constraints limited hyperparameter search; larger budgets may yield different conclusions" |
| Evaluation | "Evaluation on [metric] alone may not capture [other aspect]; future work should include [human evaluation / additional metrics]" |
| Data quality | "Potential label noise in [dataset] may affect model comparison; manual verification was not feasible at scale" |
| Bias | "The dataset may contain [demographic/selection/temporal] biases that affect model predictions" |
| Reproducibility | "Despite fixed seeds, minor non-determinism in GPU operations may cause slight metric variations" |
| Overfitting | "With N={X} samples, deep models risk overfitting; cross-validation mitigates but does not eliminate this concern" |
| Missing modalities | "Our analysis used only [text/image]; incorporating [additional modality] may improve performance" |

**Rules**:
- Be specific -- "dataset of 2,450 samples" not "small dataset"
- State the implication of each limitation
- Do NOT propose solutions (that goes in Future Directions)

### Paragraph 6: Implications & Future Directions

**Purpose**: What the findings mean and what comes next.

**Pattern**:
```
These findings have [practical/scientific/engineering] implications. [Specific
implication 1 -- what should practitioners/researchers do with this information].
[Specific implication 2 if applicable].

Future research should [direction 1 -- e.g., evaluate on larger and more diverse
datasets]. Additionally, [direction 2 -- e.g., explore few-shot or zero-shot
approaches to reduce annotation requirements]. [Direction 3 -- e.g., investigate
the feature interactions identified by tree-based models using targeted experiments].
```

**Rules**:
- At least 1 practical implication
- At least 2 future directions
- Future directions should address the limitations identified above
- If classical ML was competitive, suggest investigating when deep learning is truly necessary
- End on a constructive note

---

## Tone Rules

| Do | Don't |
|----|-------|
| "achieves comparable performance" | "destroys", "crushes", "dominates" |
| "outperforms the baseline by X%" | "is far superior to" |
| "suggests" | "proves", "demonstrates conclusively" |
| "our analysis indicates" | "we have shown definitively" |
| "consistent across model families" | "universally true" |
| Name the specific model | "advanced AI techniques" |
| "the difference is not statistically significant" | "the models are the same" |
| "these findings" | "our groundbreaking/novel/innovative results" |

## Cross-Model Synthesis Phrasing

Use these patterns in Paragraphs 1 and 4 to describe cross-model convergence:

- "The [performance pattern] was observed across all model families, from [logistic regression] to [fine-tuned transformers], suggesting the signal is robust to model complexity."
- "While [neural model] achieved the highest raw accuracy (X%), [classical baseline] was within Y percentage points, and both identified [feature Z] as most important, indicating a strong underlying signal."
- "The agreement between [model 1] and [model 2] on [finding] strengthens the evidence, while [model 3] uniquely revealed [additional insight such as nonlinear interactions]."
- "Interestingly, [simpler model] matched [complex model] on [metric], suggesting that for this dataset, the added complexity may not be justified -- a finding with practical implications for deployment."
