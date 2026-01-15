# LaTeX Report Compilation

## IDE Auto-Compile (Recommended for Cursor/VS Code)

The IDE should automatically detect changes and recompile. If not:

1. **Using VS Code/Cursor with LaTeX Workshop extension:**
   - Install "LaTeX Workshop" extension
   - It will auto-compile on save
   - PDF preview will update in the IDE

2. **Using Tasks (Built-in):**
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Tasks: Run Task"
   - Select "Auto-compile LaTeX (Watch Mode)"
   - This runs in background and updates PDF on save

## Manual Compile

```bash
./compile.sh
```

## Silent Auto-Compile (Terminal - No Popups)

```bash
./auto_compile_quiet.sh
```

This runs in the background and only updates the PDF file - no Adobe popups.

## Notes

- The PDF will be generated as `report.pdf`
- Auto-compile will update the PDF whenever you save `report.tex`
- Make sure the figure path is correct: `../cs255_cw_files/graphs/performance_comparison.pdf`
- The `.latexmkrc` file configures latexmk to work silently
