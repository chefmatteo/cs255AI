# cs255AI

Connect Game - AI Coursework Project

This project implements the minimax algorithm with and without alpha-beta pruning for the Connect game. The goal is to have a given number of pieces arranged in a single, unbroken line (horizontally, vertically, or diagonally).

## Setup

### Virtual Environment

**Note:** If you're on macOS and using Homebrew Python, you may need to use the system Python for tkinter support:
```bash
/usr/bin/python3 -m venv venv
```

Otherwise, create a virtual environment normally:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
   - On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```
   - On Windows:
   ```bash
   venv\Scripts\activate
   ```

3. Install dependencies (if using Flask frontend):
```bash
pip install -r requirements.txt
```

4. Deactivate when done:
```bash
deactivate
```

## Running the Game

### Human vs Bot (Command Line)
Play against a bot (random or minimax):
```bash
cd cs255_cw_files
python3 runGame.py
```

You can edit `runGame.py` to:
- Change board size (rows, columns)
- Change win condition (number of pieces in a line)
- Switch between random player and minimax opponent
- Enable/disable alpha-beta pruning

### Bot vs Bot
Run two bots against each other for testing and evaluation:
```bash
cd cs255_cw_files
python3 runGameTwoBots.py
```

This script uses `bots_game.py` which is optimized for bot vs bot gameplay. You can configure:
- Both players to use alpha-beta pruning (fastest)
- Both players to use regular minimax
- One player with alpha-beta, one without
- Different board configurations

Edit `runGameTwoBots.py` to customize the game setup.

### Tkinter GUI Frontend
```bash
python3 frontend_tkinter.py
```

### Flask Web Frontend (Optional)
**Note:** Requires Flask to be installed (see Setup section above).
```bash
source venv/bin/activate  # Activate virtual environment first
cd cs255_cw_files
python3 frontend_flask.py
# Then open http://localhost:5000 in your browser
```

## Project Structure

```
cs255AI/
├── cs255_cw_files/              # Core coursework implementation
│   ├── board.py                 # Board class and game logic
│   ├── game.py                  # Game class managing human vs bot gameplay
│   ├── bots_game.py             # Game class optimized for bot vs bot gameplay
│   ├── player.py                # Player class with minimax implementation
│   ├── opponent.py              # Opponent class (wrapper around Player)
│   ├── randomPlayer.py          # Random player for testing
│   ├── runGame.py               # Command line game runner (human vs bot)
│   ├── runGameTwoBots.py        # Bot vs bot game runner
│   ├── readme.txt               # Original coursework instructions
│   ├── templates/               # Report templates and plotting examples
│   │   ├── plot-example.py      # Example plotting script
│   │   └── report-template.tex  # LaTeX report template
│   └── report/                  # Report materials (PDFs)
└── requirements.txt             # Python dependencies
```

## Key Features

- **Minimax Algorithm**: Implemented in `player.py` for optimal play
- **Alpha-Beta Pruning**: Optimized minimax with pruning for better performance
- **Flexible Game Configuration**: Support for different board sizes and win conditions
- **Multiple Player Types**: 
  - Human players (via game.py)
  - Minimax bots (Player class)
  - Random players (for testing)
- **Bot vs Bot Mode**: Specialized game mode for evaluating AI performance

## Implementation Details

The minimax algorithm is implemented in `player.py`. Key methods:
- `getMove()`: Returns the best move using minimax without pruning
- `getMoveAlphaBeta()`: Returns the best move using minimax with alpha-beta pruning

The `opponent.py` class provides a convenient wrapper around the Player class for creating opponents with configurable pruning behavior.
