#!/bin/bash
# Silent auto-compile that works with IDE - no popups, just updates PDF

cd "$(dirname "$0")"

echo "Starting silent auto-compilation..."
echo "PDF will update automatically when you save report.tex"
echo "Press Ctrl+C to stop"
echo ""

# Use latexmk with continuous preview but don't open PDF
# -pvc: preview continuously (watches for changes)
# -pdf: generate PDF
# -interaction=nonstopmode: don't stop for errors
# -silent: minimal output
# -f: force mode (continue even with errors)
latexmk -pdf -pvc -interaction=nonstopmode -silent -f report.tex 2>&1 | grep -E "(Latexmk|Error|Warning|Output)" || true
