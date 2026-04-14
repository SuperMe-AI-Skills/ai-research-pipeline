# Paper Compiling: LaTeX to PDF

Compile LaTeX manuscript to PDF and resolve errors.

## Procedure

### Step 1: Pre-Flight Check

Verify all required files exist:
```bash
# Check all \input files
grep -o '\\input{[^}]*}' paper/main.tex | sed 's/\\input{//;s/}//' | while read f; do
  [ -f "paper/$f.tex" ] || echo "MISSING: paper/$f.tex"
done

# Check all figure files
grep -o '\\includegraphics\[.*\]{[^}]*}' paper/sections/*.tex | \
  grep -o '{[^}]*}' | tr -d '{}' | while read f; do
  [ -f "paper/$f" ] || echo "MISSING: paper/$f"
done

# Check bib file
[ -f "paper/references.bib" ] || echo "MISSING: paper/references.bib"
```

### Step 2: Compile

```bash
cd paper && latexmk -pdf -interaction=nonstopmode main.tex 2>&1 | tee compile.log
```

If `latexmk` not available:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Step 3: Error Diagnosis and Auto-Fix

Common errors and fixes:

| Error | Cause | Auto-Fix |
|-------|-------|----------|
| `Undefined control sequence` | Missing package or typo | Add `\usepackage{}` or fix typo |
| `File not found` | Missing figure or input | Check path, create placeholder |
| `Citation undefined` | Missing bib entry | Add to references.bib |
| `Missing $ inserted` | Math outside math mode | Wrap in `$...$` |
| `Overfull \hbox` | Line too long | Rephrase or add `\linebreak` |
| `Package not found` | Missing LaTeX package | Install via tlmgr or simplify |
| `Too many unprocessed floats` | Too many figures/tables | Add `\clearpage` |

Auto-fix loop (up to 3 iterations):
1. Parse compile.log for errors
2. Apply fix
3. Recompile
4. Check if error resolved

### Step 4: Post-Compilation Checks

```bash
# Check for unresolved references
grep -c '\?\?' paper/main.pdf  # Should be 0

# Check page count
pdfinfo paper/main.pdf | grep Pages

# Check for missing citations
grep 'Citation.*undefined' paper/main.log
```

### Step 5: Write Compile Report

```markdown
## Compilation Report

- **Status**: SUCCESS / FAILED
- **PDF**: paper/main.pdf
- **Pages**: {N} (target: {venue_limit})
- **Warnings**: {N}
- **Errors fixed**: {list}
- **Remaining issues**: {list or "none"}
```

## Key Rules

- Always compile at least twice (for cross-references)
- If using bibtex, compile sequence is: pdflatex -> bibtex -> pdflatex -> pdflatex
- Check page count against venue limit after every compilation
- Preserve compile.log for debugging
- If compilation fails after 3 auto-fix attempts, report error to user
