#!/bin/bash
# Alternative auto-compile using file watching (if fswatch is installed)
# Install fswatch with: brew install fswatch

cd "$(dirname "$0")"

if ! command -v fswatch &> /dev/null; then
    echo "Error: fswatch not found. Install with: brew install fswatch"
    echo "Or use auto_compile.sh which uses latexmk"
    exit 1
fi

echo "Watching report.tex for changes..."
echo "Press Ctrl+C to stop"
echo ""

# Watch for changes and recompile
fswatch -o report.tex | while read f; do
    echo "File changed, recompiling..."
    pdflatex -interaction=nonstopmode report.tex > /dev/null 2>&1
    pdflatex -interaction=nonstopmode report.tex > /dev/null 2>&1
    echo "✓ PDF updated: report.pdf"
    echo ""
done
