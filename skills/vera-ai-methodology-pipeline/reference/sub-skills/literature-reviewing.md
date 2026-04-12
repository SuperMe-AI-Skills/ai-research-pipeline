# Literature Reviewing: AI/ML Research

Conduct a comprehensive literature survey for: **$ARGUMENTS**

## Search Strategy

### Source Priority

1. **arXiv preprints** (primary for ML research):
   - cs.LG (Machine Learning)
   - cs.CL (Computation and Language)
   - cs.CV (Computer Vision)
   - cs.AI (Artificial Intelligence)
   - stat.ML (Machine Learning - Statistics)

2. **Top ML Conferences** (last 3 years):
   - NeurIPS, ICML, ICLR
   - ACL, EMNLP, NAACL (NLP)
   - CVPR, ECCV, ICCV (Vision)
   - AAAI, IJCAI (General AI)
   - AISTATS, UAI (Statistical ML)

3. **Journals**:
   - JMLR, TMLR, IEEE TPAMI, TACL

4. **Aggregators**:
   - Google Scholar, Semantic Scholar
   - Papers With Code (for benchmark SOTA)

5. **Local PDFs**: Check `papers/` and `literature/` in the project directory

### Search Execution

1. Use 5+ different query formulations via WebSearch
2. Read abstracts and introductions of top 15-20 papers
3. Follow citation chains: check "cited by" for seminal papers
4. Check Papers With Code for benchmark leaderboards
5. Identify the 2-3 most-cited papers in the area (anchors)

### Organization

Group papers into themes:
- **Foundational work**: Seminal papers that defined the approach
- **Recent advances**: Last 2 years, building on foundations
- **Concurrent work**: Submitted/published in the same cycle
- **Adjacent methods**: Related but different approaches
- **Benchmarks & datasets**: Standard evaluation resources

## Output Format

Write a structured literature landscape:

```markdown
# Literature Survey: {research_direction}

## Date: {timestamp}
## Query Summary: {list of search queries used}

## Landscape Overview
[3-5 paragraphs summarizing the field's current state]

## Thematic Groups

### Theme 1: {e.g., "Efficient Attention Mechanisms"}
- [Author (Year)] "{Title}" -- [1-sentence summary of contribution]
- [Author (Year)] "{Title}" -- [1-sentence summary]
- **Consensus**: [what this line of work agrees on]
- **Open questions**: [what remains unresolved]

### Theme 2: {e.g., "Self-Supervised Pre-training"}
...

### Theme 3: ...

## Open Problems & Gaps
1. {Gap 1}: [description + which papers identify it]
2. {Gap 2}: [description]
3. {Gap 3}: [description]

## Key Benchmarks
| Benchmark | Current SOTA | Method | Year |
|-----------|-------------|--------|------|
| ... | ... | ... | ... |

## References
[Full citation list in BibTeX format]
```

## Key Rules

- Focus on METHODOLOGY papers, not pure application papers (unless the application reveals new phenomena)
- Note whether results are reproduced or self-reported
- Flag papers that claim SOTA but haven't been independently verified
- Distinguish between peer-reviewed (conference/journal) and preprint-only work
- Time-box to 20 minutes maximum for methodology pipeline, 15 minutes for application pipeline
- Minimum: 10 papers. Target: 15-25 papers.
