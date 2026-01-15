#!/bin/bash
# Simple compilation script for report.tex

cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode report.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode report.tex > /dev/null 2>&1
echo "✓ PDF compiled: report.pdf"
