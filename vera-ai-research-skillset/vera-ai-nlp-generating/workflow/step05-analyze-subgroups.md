# 05 --- Analyze Subgroups

## Executor: Main Agent (code generation)

## Data In: Code + prose from 04-run-additional-models.md

## Skip if no subgroup variable specified. Pass through to 06-fit-advanced-models.md.

## Generate code for PART 4: Subgroup Analysis

### 5A: Subgroup Definition
Subgroups can be defined by:
- Metadata variable (e.g., source, author type, domain)
- Text property (e.g., text length bins: short/medium/long)
- AI confidence bins (e.g., high/medium/low confidence from AI detector)
- Custom user-defined variable

### 5B: Per-Subgroup Evaluation
For the best ML model from step 04 (or baseline LogReg):
1. Evaluate on each subgroup subset
2. Report F1 + AUC per subgroup with 95% CIs
3. Flag subgroups with n < 20 (unreliable estimates)

### 5C: Subgroup Comparison
1. **Performance disparity table**: F1/AUC per subgroup side-by-side
2. **Worst-subgroup analysis**: identify subgroup with lowest F1
3. **Fairness metrics** (if applicable): equalized odds, demographic parity

### 5D: Visualizations
1. **Grouped bar chart of F1 by subgroup** → `plot_04_subgroup_f1.png`
2. **Per-subgroup confusion matrices** (small multiples) → `plot_04_subgroup_cm.png`

### Quality: Apply structure variation
- Bar chart: alternate horizontal vs vertical, legend position
- Vary inline vs standalone subgroup performance reporting
- Rotate framing: "performance was consistent" / "notable disparities emerged" / "varied by subgroup"

## Validation Checkpoint

- [ ] Subgroups defined and sample sizes reported
- [ ] Per-subgroup F1 + AUC with CIs
- [ ] Small subgroups flagged (n < 20)
- [ ] Performance disparity table generated
- [ ] Worst-subgroup identified
- [ ] Subgroup visualizations generated

## Data Out -> 06-fit-advanced-models.md

```
subgroup_code_py: [PART 4 Python code]
methods_para_subgroup: [methods paragraph prose]
results_para_subgroup: [results paragraph prose]
plots: [list of new plot filenames]
```
