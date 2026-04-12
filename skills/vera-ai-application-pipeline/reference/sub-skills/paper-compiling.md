# Paper Compiling: LaTeX to PDF (Application Pipeline)

Compile LaTeX manuscript to PDF and resolve errors.

## Procedure

### Step 1: Pre-Flight Check

```bash
# Verify all \input files exist
grep -o '\\input{[^}]*}' paper/main.tex | sed 's/\\input{//;s/}//' | while read f; do
  [ -f "paper/$f.tex" ] || echo "MISSING: paper/$f.tex"
done

# Verify all figures exist
grep -o '\\includegraphics\[.*\]{[^}]*}' paper/sections/*.tex | \
  grep -o '{[^}]*}' | tr -d '{}' | while read f; do
  [ -f "paper/$f" ] || echo "MISSING: paper/$f"
done

# Verify bibliography
[ -f "paper/references.bib" ] || echo "MISSING: paper/references.bib"
```

### Step 2: Compile

```bash
cd paper && latexmk -pdf -interaction=nonstopmode main.tex 2>&1 | tee compile.log
```

Fallback (no latexmk):
```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### Step 3: Auto-Fix Errors (up to 3 iterations)

| Error | Fix |
|-------|-----|
| Undefined control sequence | Add missing `\usepackage` or fix typo |
| File not found | Check path, create placeholder |
| Citation undefined | Add to references.bib |
| Missing $ | Wrap in math mode |
| Overfull hbox | Rephrase or allow slight overflow |

### Step 4: Post-Compilation Checks

- No `??` references in PDF
- Page count within venue limit
- All citations resolve
- All figures render correctly

### Step 5: Write Compile Report

```markdown
## Compilation Report
- **Status**: SUCCESS / FAILED
- **PDF**: paper/main.pdf
- **Pages**: {N} (limit: {venue_limit})
- **Warnings**: {N}
- **Remaining issues**: {list or "none"}
```

## Key Rules

- Compile at least twice for cross-references
- With bibtex: pdflatex -> bibtex -> pdflatex -> pdflatex
- Check page count after every compilation
- If fails after 3 attempts, report to user with specific error
