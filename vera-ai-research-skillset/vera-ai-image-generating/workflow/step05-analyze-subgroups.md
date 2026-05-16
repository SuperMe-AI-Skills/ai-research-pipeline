# 05 --- Analyze Subgroups

## Executor: Main Agent (code generation)

## Data In: Code + prose from 04-run-additional-models.md

## Skip if no subgroup variable specified. Pass through to 06-fit-advanced-models.md.

## Generate code for PART 4: Subgroup Analysis

### 5A: Subgroup Definition
Subgroups can be defined by:
- Metadata variable (e.g., imaging device, acquisition site, patient demographics)
- Image property (e.g., image resolution bins: low/medium/high, aspect ratio groups)
- Brightness/contrast bins (e.g., dark/normal/bright based on mean pixel intensity)
- Medical imaging metadata (e.g., scanner manufacturer, slice thickness, body region)
- Custom user-defined variable

### 5B: Per-Subgroup Evaluation
For the best CNN model from step 04 (or baseline model):
1. Evaluate on each subgroup subset
2. Report F1 + AUC per subgroup with 95% CIs
3. Flag subgroups with n < 20 (unreliable estimates)

### 5C: Subgroup Comparison
1. **Performance disparity table**: F1/AUC per subgroup side-by-side
2. **Worst-subgroup analysis**: identify subgroup with lowest F1
3. **Fairness metrics** (if applicable): equalized odds, demographic parity
4. **Domain shift analysis**: flag if subgroup performance drop exceeds 0.05 F1

### 5D: Visualizations
1. **Grouped bar chart of F1 by subgroup** -> `plot_04_subgroup_f1.png`
2. **Per-subgroup confusion matrices** (small multiples) -> `plot_04_subgroup_cm.png`
3. **GradCAM comparison across subgroups** (if image property subgroups) -> `plot_04_subgroup_gradcam.png`

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
