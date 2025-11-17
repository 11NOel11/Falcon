# FALCON v5 + Muon Analysis - LaTeX Paper

This directory contains the complete LaTeX source for the combined FALCON v5 and Muon optimizer analysis paper.

## Contents

```
latex_paper/
├── main.tex                 # Main LaTeX document
├── references.bib           # Bibliography file
├── README.md               # This file
├── figures/                # All figures (14 PNG files at 300 dpi)
│   ├── fig_top1_vs_time.png
│   ├── fig_time_to_85.png
│   ├── fig_fixed_time_10min.png
│   ├── fig_data_efficiency.png
│   ├── fig_real_image_filtering.png
│   ├── fig_frequency_masks.png
│   ├── fig_computational_breakdown.png
│   ├── fig_adaptive_schedules.png
│   ├── fig_ema_averaging.png
│   ├── fig_progressive_filtering.png
│   ├── fig_mask_sharing.png
│   ├── fig_architecture_comparison.png
│   ├── fig_robustness_noise.png
│   └── fig_frequency_filtering_demo.png
└── sections/               # Individual section files
    ├── 01_introduction.tex
    ├── 02_related_work.tex
    ├── 03_method_falcon.tex
    ├── 04_method_muon.tex
    ├── 05_experimental_setup.tex
    ├── 06_results.tex
    ├── 07_analysis.tex
    ├── 08_ablation.tex
    └── 09_conclusion.tex
```

## Compilation Instructions

### Required LaTeX Distribution

This paper uses the **CVPR 2026** document class. You need a full TeX distribution installed:

- **Linux**: Install `texlive-full`
- **macOS**: Install MacTeX
- **Windows**: Install MiKTeX or TeX Live

### Option 1: Using pdflatex (Recommended)

```bash
cd latex_paper

# First pass: generate aux files
pdflatex main.tex

# Second pass: process bibliography
bibtex main

# Third pass: resolve citations
pdflatex main.tex

# Fourth pass: resolve cross-references
pdflatex main.tex
```

This produces `main.pdf` with all citations and cross-references resolved.

### Option 2: Using latexmk (Automated)

If you have `latexmk` installed (included with most distributions):

```bash
cd latex_paper
latexmk -pdf main.tex
```

This automatically runs all necessary passes.

### Option 3: Overleaf (Online)

1. Compress all files: `zip -r falcon_paper.zip latex_paper/`
2. Upload to [Overleaf](https://www.overleaf.com/)
3. Set compiler to **pdfLaTeX**
4. Click "Recompile"

## Document Structure

The paper is organized into 9 sections:

1. **Introduction** - Motivation, contributions, key findings
2. **Related Work** - Adaptive optimization, second-order methods, frequency-domain analysis
3. **Method: FALCON v5** - Architecture, mathematical formulation, implementation
4. **Method: Muon** - Hybrid design, computational analysis
5. **Experimental Setup** - Datasets, models, training configuration
6. **Results** - Full training, convergence, fixed-time, data efficiency
7. **Analysis** - Why parity not superiority, computational overhead, practical recommendations
8. **Ablation Studies** - Component analysis, hyperparameter sensitivity
9. **Conclusion** - Summary, limitations, future work

## Key Features

- **CVPR Conference Format**: Two-column layout, proper citations, IEEE-style references
- **35+ Equations**: Fully typeset mathematical formulations
- **18+ Tables**: Comprehensive experimental results
- **14 Figures**: High-resolution (300 dpi) visualizations
- **40+ References**: Complete bibliography
- **~10,400 Words**: Combined content from both papers

## Customization

### Change Title/Authors

Edit `main.tex` around line 20:

```latex
\title{Your New Title}

\author{
    First Author\thanks{Affiliation} \\
    {\tt\small first.author@email.com} \\
    \and
    Second Author \\
    {\tt\small second.author@email.com}
}
```

### Add/Remove Sections

Edit `main.tex` around line 50-60 to comment out or add `\input{}` commands:

```latex
\input{sections/01_introduction}
\input{sections/02_related_work}
% \input{sections/03_method_falcon}  % Comment out to exclude
\input{sections/04_method_muon}
```

### Modify Figure Sizes

In section files, adjust `width` parameter:

```latex
\includegraphics[width=0.48\columnwidth]{figures/fig_top1_vs_time}
```

Options: `0.48\columnwidth` (half-column), `0.98\columnwidth` (full-column), `\textwidth` (two-column)

### Change Citation Style

Edit `main.tex` around line 75:

```latex
\bibliographystyle{ieee}      % Current style
% \bibliographystyle{plain}   % Alternative: alphabetical
% \bibliographystyle{abbrv}   % Alternative: abbreviated
```

## Common Issues

### Missing CVPR Class File

If you get `cvpr.cls not found`:

1. Download from CVPR website or use standard article class:
   ```latex
   \documentclass[10pt,twocolumn]{article}
   ```

2. Install manually:
   ```bash
   wget http://www.cvpr.org/cvpr.cls
   cp cvpr.cls /usr/share/texlive/texmf-dist/tex/latex/cvpr/
   sudo mktexlsr
   ```

### Missing Packages

If compilation fails with missing packages:

```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-extra texlive-fonts-extra

# macOS (with Homebrew)
brew install --cask mactex

# Or install individual packages via tlmgr
tlmgr install booktabs algorithm algorithmicx
```

### Bibliography Not Showing

Ensure you run `bibtex` between `pdflatex` passes:

```bash
pdflatex main.tex
bibtex main       # Note: no .tex extension
pdflatex main.tex
pdflatex main.tex
```

### Figures Not Displaying

Check that:
1. All PNG files are in `figures/` directory
2. Paths in `\includegraphics{}` match filenames exactly (case-sensitive)
3. No spaces in filenames

## Statistics

- **Lines of LaTeX**: ~4,200
- **Equations**: 35+
- **Tables**: 18+
- **Figures**: 14
- **References**: 40+
- **Estimated Pages**: 12-14 (CVPR two-column format)

## Version Control

Original markdown papers:
- `../paper_stuff/CVPR_PAPER_FALCON_V5.md` (5,600 words)
- `../paper_stuff/CVPR_PAPER_MUON_ANALYSIS.md` (4,800 words)

Figures source:
- `../paper_stuff/*.png` (14 files)

## License

See `../LICENSE` for licensing information.

## Contact

For questions about this LaTeX conversion or paper content, refer to the main project README.

## Troubleshooting

### Quick Syntax Check

```bash
# Check for LaTeX syntax errors without full compilation
chktex main.tex
```

### Clean Build Files

```bash
# Remove auxiliary files
rm -f *.aux *.bbl *.blg *.log *.out *.toc sections/*.aux

# Full clean (keeps only source)
latexmk -C
```

### Reduce File Size

If PDF is too large:

```bash
# Compress PDF (requires Ghostscript)
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
   -dNOPAUSE -dQUIET -dBATCH -sOutputFile=main_compressed.pdf main.pdf
```

## Acknowledgments

LaTeX conversion performed automatically. All content faithfully reproduced from original markdown papers with proper formatting for academic publication.

---

**Last Updated**: 2025  
**Compiler Tested**: pdfTeX 3.141592653-2.6-1.40.24 (TeX Live 2022)  
**Format**: CVPR Conference Paper (Two-Column)
