# Project Structure

This document describes the clean, organized structure of the CS255 AI Coursework project.

## Directory Hierarchy

```
cs255AI/
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── spec.pdf                     # Coursework specification
│
├── frontend_tkinter.py          # Tkinter GUI frontend (standalone)
│
└── cs255_cw_files/              # Core coursework implementation
    ├── board.py                 # Board class and game logic
    ├── game.py                  # Game class managing gameplay
    ├── player.py                # Player class (minimax implementation)
    ├── randomPlayer.py          # Random player for testing
    ├── runGame.py               # Command-line game runner
    ├── readme.txt               # Coursework instructions
    │
    ├── frontend_flask.py        # Flask web frontend (optional)
    ├── templates/                # Flask templates directory (empty, for future use)
    │
    └── report/                   # Report materials
        ├── report-template.tex  # LaTeX report template
        ├── report-template.pdf  # PDF version of template
        ├── figure_example.pdf   # Example figure
        └── plot-example.py      # Example plotting script
```

## File Organization Logic

### Core Game Files (`cs255_cw_files/`)
All core game implementation files are kept together in `cs255_cw_files/`:
- **board.py**: Board representation and game state management
- **game.py**: Game orchestration and turn management
- **player.py**: AI player with minimax algorithm (your implementation)
- **randomPlayer.py**: Random opponent for testing
- **runGame.py**: Simple command-line interface to run games

### Frontend Files
Two different frontend implementations are available:
- **frontend_tkinter.py** (root): Standalone Tkinter GUI that imports from `cs255_cw_files/`
- **frontend_flask.py** (in cs255_cw_files): Flask web application frontend

### Report Files
All report-related materials are consolidated in `cs255_cw_files/report/`:
- LaTeX template and PDF
- Example figures and plotting scripts

## Running the Project

### Command Line Game
```bash
cd cs255_cw_files
python3 runGame.py
```

### Tkinter GUI
```bash
python3 frontend_tkinter.py
```

### Flask Web App
```bash
cd cs255_cw_files
python3 frontend_flask.py
```

## Notes for Graders

1. **Core Implementation**: All coursework implementation is in `cs255_cw_files/`
2. **Main File**: `player.py` contains the minimax algorithm implementation
3. **Testing**: Use `runGame.py` to test the AI implementation
4. **Report**: Report materials and template are in `cs255_cw_files/report/`
5. **Frontends**: Additional frontend implementations are provided for convenience but are not part of the core coursework
