#!/bin/bash
# Auto-compile LaTeX on file changes using latexmk

cd "$(dirname "$0")"

echo "Starting auto-compilation for report.tex..."
echo "Press Ctrl+C to stop"
echo ""

# Use latexmk with continuous preview mode
# -pvc: preview continuously (watches for changes)
# -pdf: generate PDF
# -interaction=nonstopmode: don't stop for errors
latexmk -pdf -pvc -interaction=nonstopmode report.tex
