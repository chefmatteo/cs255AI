# CS255 AI: Adversarial Search - Connect Game

A comprehensive implementation and evaluation of minimax and alpha-beta pruning algorithms for the Connect game. This project includes benchmarking tools, performance analysis, and a detailed research report comparing algorithm performance across diverse board configurations.

## Overview

This coursework project implements adversarial search algorithms (minimax with and without alpha-beta pruning) for Connect, a gravity-based board game where players aim to form a line of a specified length. The project includes:

- **Core game implementation** with flexible board sizes and win conditions
- **Minimax algorithm** with optional alpha-beta pruning
- **Comprehensive benchmarking suite** for performance analysis
- **Statistical analysis** across multiple configurations
- **Research report** analyzing algorithm scaling behavior and complexity

## Project Structure

```
cs255AI/
├── cs255_cw_files/              # Core implementation
│   ├── board.py                  # Board class and game logic
│   ├── game.py                   # Human vs bot gameplay
│   ├── bots_game.py              # Optimized bot vs bot gameplay
│   ├── player.py                 # Minimax implementation (with/without pruning)
│   ├── opponent.py               # Opponent wrapper class
│   ├── randomPlayer.py           # Random player for testing
│   ├── runGame.py                # Command line game runner
│   ├── runGameTwoBots.py         # Bot vs bot game runner
│   ├── graphs/                   # Benchmarking and visualization
│   │   ├── benchmark.py          # Comprehensive benchmarking suite
│   │   ├── run_benchmark.py      # Benchmark execution script
│   │   ├── benchmark_results.csv # Performance data
│   │   └── performance_comparison.pdf/png  # Visualization outputs
│   ├── templates/                # Report templates
│   │   ├── report-template.tex   # LaTeX template
│   │   └── plot-example.py       # Plotting examples
│   └── report-resources/         # Additional report materials
├── report/                       # Research report
│   ├── report.tex                # Main LaTeX document
│   ├── report.pdf                # Compiled report
│   ├── compile.sh                # Manual compilation script
│   ├── auto_compile.sh           # Auto-compile script
│   └── watch_compile.sh          # Watch mode compilation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup

### Prerequisites

- Python 3.7 or higher
- LaTeX distribution (for report compilation): TeX Live, MacTeX, or MiKTeX

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chefmatteo/cs255AI.git
   cd cs255AI
   ```

2. **Create virtual environment:**
   
   On macOS with Homebrew Python (for tkinter support):
   ```bash
   /usr/bin/python3 -m venv venv
   ```
   
   Otherwise:
   ```bash
   python3 -m venv venv
   ```

3. **Activate virtual environment:**
   - macOS/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Playing the Game

**Human vs Bot (Command Line):**
```bash
cd cs255_cw_files
python3 runGame.py
```

Edit `runGame.py` to configure:
- Board size (rows × columns)
- Win condition (`winNum` - number of pieces in a line)
- Algorithm choice (minimax with/without alpha-beta pruning)
- Search depth

**Bot vs Bot:**
```bash
cd cs255_cw_files
python3 runGameTwoBots.py
```

Optimized for automated gameplay and testing. Configure both players' algorithms independently.

### Benchmarking and Performance Analysis

Run comprehensive benchmarks across multiple configurations:

```bash
cd cs255_cw_files/graphs
python3 run_benchmark.py
```

The benchmarking suite (`benchmark.py`) evaluates:
- Multiple board sizes (4×4, 5×5, 5×6, 6×7, 8×8)
- Different win conditions (`winNum` values)
- Various search depths (3, 4, 5)
- Performance metrics:
  - Node expansion counts
  - Execution time
  - Pruning efficiency
  - Effective branching factor
  - Time speedup factors

Results are exported to CSV files and visualized in `performance_comparison.pdf`.

### Report Compilation

The research report analyzes algorithm performance across 18 benchmark configurations. To compile:

**Manual compilation:**
```bash
cd report
./compile.sh
```

**Auto-compile (watch mode):**
```bash
cd report
./auto_compile.sh
```

**Silent auto-compile (no popups):**
```bash
cd report
./auto_compile_quiet.sh
```

The compiled PDF will be available as `report/report.pdf`.

## Key Features

### Algorithm Implementation

- **Minimax Algorithm**: Optimal decision-making for adversarial games
- **Alpha-Beta Pruning**: Optimized minimax with up to 99.6% node reduction
- **Configurable Search Depth**: Adjustable lookahead for performance/quality trade-offs
- **Flexible Game Configuration**: Support for arbitrary board sizes and win conditions

### Performance Analysis

The project includes extensive benchmarking revealing:

- **Super-linear scaling**: Pruning efficiency increases faster than problem size (82% to 99.6% reduction)
- **Exponential speedup**: Time improvements from 103× to 3,956× depending on configuration
- **Effective branching factor analysis**: Quantifies actual vs. theoretical branching behavior
- **Statistical evaluation**: Mean and standard deviation across 18 configurations

### Research Findings

Key insights from the analysis:

1. **Pruning efficiency scales logarithmically** with board area
2. **Alpha-beta becomes essential** for boards >5×5 at depth >3
3. **Win condition (`winNum`) significantly impacts** algorithm performance
4. **Deeper searches show higher pruning rates** on larger boards
5. **Complexity ratio decreases super-linearly** with board size and depth

## Implementation Details

### Core Classes

- **`Board`**: Manages game state, move validation, and win detection
- **`Player`**: Implements minimax with methods:
  - `getMove()`: Minimax without pruning
  - `getMoveAlphaBeta()`: Minimax with alpha-beta pruning
- **`Game`**: Manages human vs bot gameplay
- **`BotsGame`**: Optimized for bot vs bot scenarios

### Algorithm Configuration

Edit game runner files to customize:
- Board dimensions (rows, columns)
- Win condition (`winNum`)
- Search depth
- Pruning enabled/disabled
- Player types (human, minimax, random)

## Dependencies

- **Core**: Python standard library (no external dependencies required)
- **Benchmarking**: numpy, pandas, matplotlib, seaborn
- **Optional**: Flask (for web frontend, if implemented)

## Results Summary

Based on comprehensive benchmarking across 18 configurations:

- **Average node reduction**: 95.5% ± 4.7%
- **Average time speedup**: 1,255.8× ± 1,560.9×
- **Average pruning rate**: 22.6% ± 5.9%
- **Average EBF reduction**: 54.1% ± 10.0%

See `report/report.pdf` for detailed analysis and visualizations.

## License

This project is part of CS255 coursework. See the original coursework specification for details.

## Author

21052135 - Adversarial Search Tasks
