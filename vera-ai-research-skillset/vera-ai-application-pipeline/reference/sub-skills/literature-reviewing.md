# Literature Reviewing: AI/ML Application Research

Conduct a focused literature survey for applied AI/ML research: **$ARGUMENTS**

## Search Strategy

### Source Priority

1. **arXiv preprints** (primary for ML):
   - cs.LG, cs.CL, cs.CV, cs.AI, stat.ML
   - Domain-specific: cs.IR (retrieval), cs.SD (sound), q-bio (computational biology)

2. **Top Conferences** (last 3 years):
   - General ML: NeurIPS, ICML, ICLR
   - NLP: ACL, EMNLP, NAACL
   - Vision: CVPR, ECCV, ICCV
   - Domain: KDD, WWW, MICCAI, CHIL (depending on application area)

3. **Journals**: JMLR, TMLR, IEEE TPAMI, TACL, Nature Machine Intelligence

4. **Aggregators**: Google Scholar, Semantic Scholar, Papers With Code

### Search Execution

1. Use 5+ different query formulations via WebSearch
2. Read abstracts of top 15-20 papers
3. Check Papers With Code for benchmark state-of-the-art on relevant tasks
4. Identify the 2-3 most-cited papers in the area

### Organization

Group papers into:
- **Standard approaches**: What most papers use for this task
- **State-of-the-art**: Best-performing methods
- **Classical baselines**: Non-DL methods still commonly reported
- **Domain-specific**: Methods tailored to the application area

## Output Format

```markdown
# Literature Survey: {research_question}

## Date: {timestamp}

## How Others Have Analyzed Similar Data

### Common Models/Methods
- [Model 1]: Used in [cite] -- [performance summary]
- [Model 2]: Used in [cite] -- [performance summary]

### Reporting Conventions
- Standard metrics: [list]
- Expected baselines: [list]
- Analysis expectations: [ablation, error analysis, etc.]

### Gaps in Existing Analyses
- [Gap 1]
- [Gap 2]

### Key References
- [Author (Year)]: [1-sentence relevance]
... (target 15-25 references)

## References
[BibTeX format]
```

## Key Rules

- Time-box to 15 minutes for application pipeline
- Focus on what models/methods WORK for this task, not theoretical novelty
- Note standard metrics and baselines expected by reviewers
- Minimum 10 papers, target 15-25
- Include both deep learning AND classical ML baselines from literature
