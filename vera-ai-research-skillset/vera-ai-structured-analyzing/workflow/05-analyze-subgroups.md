# 05 --- Analyze Subgroups

## Executor: Main Agent (code generation)

## Data In: Code + prose from 04-run-additional-models.md

## Skip if no subgroup variable specified. Pass through to 06-fit-advanced-models.md.

## Generate code for PART 4: Subgroup Analysis

### 5A: Subgroup Definition
Subgroups can be defined by:
- Categorical variable (e.g., region, sex, treatment arm, facility type)
- Binned continuous variable (e.g., age bins: young/middle/old, income quartiles)
- Derived variable (e.g., missing data pattern, outlier flag)
- Custom user-defined variable

### 5B: Per-Subgroup Evaluation
For the best ML model from step 04 (or baseline LogReg):
1. Evaluate on each subgroup subset
2. **Classification**: report F1 + AUC per subgroup with 95% CIs
3. **Regression**: report RMSE + R2 per subgroup with 95% CIs
4. Flag subgroups with n < 20 (unreliable estimates)

### 5C: Subgroup Comparison
1. **Performance disparity table**: metrics per subgroup side-by-side
2. **Worst-subgroup analysis**: identify subgroup with lowest F1 (classification) or highest RMSE (regression)
3. **Fairness metrics** (if applicable, classification only): equalized odds, demographic parity

### 5D: Visualizations
1. **Grouped bar chart of primary metric by subgroup** -> `plot_04_subgroup_metric.png`
2. **Per-subgroup confusion matrices** (classification, small multiples) -> `plot_04_subgroup_cm.png`
3. **Per-subgroup residual plots** (regression) -> `plot_04_subgroup_residuals.png`

### Quality: Apply structure variation
- Bar chart: alternate horizontal vs vertical, legend position
- Vary inline vs standalone subgroup performance reporting
- Rotate framing: "performance was consistent" / "notable disparities emerged" / "varied by subgroup"

## Validation Checkpoint

- [ ] Subgroups defined and sample sizes reported
- [ ] Per-subgroup metrics with CIs (F1+AUC or RMSE+R2)
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
