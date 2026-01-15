# LaTeXmk configuration for auto-compilation
# This file configures latexmk to work silently in the background

$pdf_mode = 1;
$postscript_mode = 0;
$dvi_mode = 0;

# Don't open PDF automatically
$preview_continuous_mode = 0;

# Quiet mode (less output)
$silent = 1;

# Clean up auxiliary files
$clean_ext = 'bbl synctex.gz';

# Use pdflatex
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# Output directory
$out_dir = '.';
