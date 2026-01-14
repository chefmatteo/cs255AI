# cs255AI

Connect Game - AI Coursework Project

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

### Command Line Version
```bash
cd cs255_cw_files
python3 runGame.py
```

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
├── frontend_tkinter.py          # Tkinter GUI frontend
├── cs255_cw_files/              # Core coursework implementation
│   ├── board.py                 # Board class and game logic
│   ├── game.py                  # Game class managing gameplay
│   ├── player.py                # Player class (implement minimax here)
│   ├── randomPlayer.py          # Random player for testing
│   ├── runGame.py               # Command line game runner
│   ├── frontend_flask.py        # Flask web frontend (optional)
│   └── report/                  # Report materials and templates
└── spec.pdf                     # Coursework specification
```

For detailed structure information, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).
