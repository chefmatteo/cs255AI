"""
Connect 4 Algorithm Performance Benchmarking
Compares MinMax (without pruning) vs Alpha-Beta (with pruning) algorithms
Measures 7 core performance metrics across different board sizes
"""

import sys
import os
import time
import tracemalloc
import math
from pathlib import Path

# Add parent directory to path to import game modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import board
import player
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class BenchmarkPlayer(player.Player):
    """Extended Player class with fixed depth for benchmarking"""
    
    def __init__(self, name, fixed_depth=None):
        super().__init__(name)
        self.fixed_depth = fixed_depth
    
    def _calculateMaxDepth(self, board):
        """Override to use fixed depth if provided"""
        if self.fixed_depth is not None:
            return self.fixed_depth
        return super()._calculateMaxDepth(board)


class ConnectFourBenchmark:
    """Connect 4 Algorithm Performance Benchmarking Class"""
    
    def __init__(self, output_dir=None):
        """
        Initialize benchmark class
        
        Args:
            output_dir: Directory to save results and graphs (default: ./graphs)
        """
        if output_dir is None:
            self.output_dir = Path(__file__).parent
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = []
        
    def measure_nodes_visited(self, board_sizes, depth=4):
        """
        Measure nodes visited for different board sizes
        
        Returns: dict with board_size as key and metrics as value
        """
        print("\n" + "="*60)
        print("Metric 1: Nodes Visited")
        print("="*60)
        
        results = {}
        
        for rows, cols in board_sizes:
            print(f"\nTesting board size: {rows}x{cols} (depth={depth})")
            game_board = board.Board(rows, cols, min(4, min(rows, cols)))
            
            # Test MinMax (without pruning)
            ai_minmax = BenchmarkPlayer("X", fixed_depth=depth)
            start_time = time.time()
            move_minmax = ai_minmax.getMove(game_board.copy())
            time_minmax = time.time() - start_time
            
            # Test Alpha-Beta (with pruning)
            # Create fresh board for Alpha-Beta test
            game_board_ab = board.Board(rows, cols, min(4, min(rows, cols)))
            ai_alphabeta = BenchmarkPlayer("X", fixed_depth=depth)
            start_time = time.time()
            move_alphabeta = ai_alphabeta.getMoveAlphaBeta(game_board_ab.copy())
            time_alphabeta = time.time() - start_time
            
            nodes_minmax = ai_minmax.numExpanded
            nodes_alphabeta = ai_alphabeta.numExpanded
            nodes_reduction = 1 - (nodes_alphabeta / nodes_minmax) if nodes_minmax > 0 else 0
            
            results[(rows, cols)] = {
                'minmax_nodes': nodes_minmax,
                'alphabeta_nodes': nodes_alphabeta,
                'nodes_reduction': nodes_reduction,
                'time_minmax': time_minmax,
                'time_alphabeta': time_alphabeta,
                'pruning_count': ai_alphabeta.numPruned
            }
            
            print(f"  MinMax nodes: {nodes_minmax:,}")
            print(f"  Alpha-Beta nodes: {nodes_alphabeta:,}")
            print(f"  Reduction: {nodes_reduction*100:.1f}%")
            print(f"  Pruning count: {ai_alphabeta.numPruned:,}")
        
        return results
    
    def measure_search_time(self, board_sizes, depth=4, num_trials=3):
        """
        Measure search time for different board sizes
        
        Returns: list of dicts with time metrics
        """
        print("\n" + "="*60)
        print("Metric 2: Search Time")
        print("="*60)
        
        time_results = []
        
        for rows, cols in board_sizes:
            print(f"\nTesting board size: {rows}x{cols} ({num_trials} trials)")
            game_board = board.Board(rows, cols, min(4, min(rows, cols)))
            
            minmax_times = []
            alphabeta_times = []
            
            for trial in range(num_trials):
                # Test MinMax
                ai_minmax = BenchmarkPlayer("X", fixed_depth=depth)
                start = time.time()
                ai_minmax.getMove(game_board.copy())
                minmax_times.append(time.time() - start)
                
                # Test Alpha-Beta
                ai_ab = BenchmarkPlayer("X", fixed_depth=depth)
                start = time.time()
                ai_ab.getMoveAlphaBeta(game_board.copy())
                alphabeta_times.append(time.time() - start)
            
            avg_minmax = np.mean(minmax_times)
            avg_alphabeta = np.mean(alphabeta_times)
            speedup = avg_minmax / avg_alphabeta if avg_alphabeta > 0 else 0
            
            time_results.append({
                'board_size': f"{rows}x{cols}",
                'rows': rows,
                'cols': cols,
                'minmax_avg_time': avg_minmax,
                'alphabeta_avg_time': avg_alphabeta,
                'time_speedup': speedup,
                'minmax_std': np.std(minmax_times),
                'alphabeta_std': np.std(alphabeta_times)
            })
            
            print(f"  MinMax avg time: {avg_minmax:.4f}s (±{np.std(minmax_times):.4f})")
            print(f"  Alpha-Beta avg time: {avg_alphabeta:.4f}s (±{np.std(alphabeta_times):.4f})")
            print(f"  Speedup: {speedup:.2f}x")
        
        return time_results
    
    def calculate_branching_factor(self, board_sizes, depth=3):
        """
        Calculate effective branching factor
        
        Returns: list of dicts with branching factor metrics
        """
        print("\n" + "="*60)
        print("Metric 3: Effective Branching Factor")
        print("="*60)
        
        branching_data = []
        
        for rows, cols in board_sizes:
            print(f"\nTesting board size: {rows}x{cols} (depth={depth})")
            game_board = board.Board(rows, cols, min(4, min(rows, cols)))
            
            theoretical_bf = cols
            
            # Test MinMax
            ai_minmax = BenchmarkPlayer("X", fixed_depth=depth)
            ai_minmax.getMove(game_board.copy())
            nodes_minmax = ai_minmax.numExpanded
            
            # Test Alpha-Beta
            ai_ab = BenchmarkPlayer("X", fixed_depth=depth)
            ai_ab.getMoveAlphaBeta(game_board.copy())
            nodes_ab = ai_ab.numExpanded
            
            # Effective branching factor: (nodes)^(1/depth)
            effective_bf_minmax = nodes_minmax ** (1.0/depth) if nodes_minmax > 0 else 0
            effective_bf_ab = nodes_ab ** (1.0/depth) if nodes_ab > 0 else 0
            branching_reduction = 1 - (effective_bf_ab / effective_bf_minmax) if effective_bf_minmax > 0 else 0
            
            branching_data.append({
                'board_size': f"{rows}x{cols}",
                'rows': rows,
                'cols': cols,
                'theoretical_bf': theoretical_bf,
                'effective_bf_minmax': effective_bf_minmax,
                'effective_bf_alphabeta': effective_bf_ab,
                'branching_reduction': branching_reduction
            })
            
            print(f"  Theoretical BF: {theoretical_bf}")
            print(f"  Effective BF (MinMax): {effective_bf_minmax:.2f}")
            print(f"  Effective BF (Alpha-Beta): {effective_bf_ab:.2f}")
            print(f"  Reduction: {branching_reduction*100:.1f}%")
        
        return branching_data
    
    def measure_pruning_efficiency(self, board_sizes, depths=[2, 3, 4]):
        """
        Measure pruning efficiency at different depths
        
        Returns: list of dicts with pruning metrics
        """
        print("\n" + "="*60)
        print("Metric 4: Pruning Efficiency")
        print("="*60)
        
        pruning_results = []
        
        for rows, cols in board_sizes:
            for depth in depths:
                print(f"\nTesting {rows}x{cols} at depth {depth}")
                game_board = board.Board(rows, cols, min(4, min(rows, cols)))
                
                # Get Alpha-Beta statistics
                ai_ab = BenchmarkPlayer("X", fixed_depth=depth)
                ai_ab.getMoveAlphaBeta(game_board.copy())
                
                nodes_visited = ai_ab.numExpanded
                pruning_count = ai_ab.numPruned
                
                # Calculate pruning rate
                # Estimate total nodes without pruning (rough estimate)
                theoretical_nodes = cols ** depth if depth <= 6 else float('inf')
                if theoretical_nodes != float('inf'):
                    pruning_rate = pruning_count / (nodes_visited + pruning_count) if (nodes_visited + pruning_count) > 0 else 0
                else:
                    pruning_rate = pruning_count / nodes_visited if nodes_visited > 0 else 0
                
                pruning_results.append({
                    'rows': rows,
                    'cols': cols,
                    'board_size': f"{rows}x{cols}",
                    'depth': depth,
                    'pruning_rate': pruning_rate,
                    'nodes_visited': nodes_visited,
                    'pruning_count': pruning_count
                })
                
                print(f"  Nodes visited: {nodes_visited:,}")
                print(f"  Pruning count: {pruning_count:,}")
                print(f"  Pruning rate: {pruning_rate*100:.1f}%")
        
        return pruning_results
    
    def measure_decision_quality(self, board_sizes, num_games=20, depth=4):
        """
        Measure decision quality through self-play comparison
        
        Note: This is a simplified version. Full implementation would require
        running complete games between algorithms.
        
        Returns: list of dicts with quality metrics
        """
        print("\n" + "="*60)
        print("Metric 5: Decision Quality (Simplified)")
        print("="*60)
        print("Note: Full decision quality requires complete game simulations")
        print("This metric compares move selection consistency")
        
        quality_results = []
        
        for rows, cols in board_sizes:
            print(f"\nTesting board size: {rows}x{cols}")
            game_board = board.Board(rows, cols, min(4, min(rows, cols)))
            
            # Test multiple initial positions
            moves_agree = 0
            moves_total = 0
            
            for _ in range(num_games):
                # Create fresh board
                test_board = board.Board(rows, cols, min(4, min(rows, cols)))
                
                # Get moves from both algorithms
                ai_minmax = BenchmarkPlayer("X", fixed_depth=depth)
                move_minmax = ai_minmax.getMove(test_board.copy())
                
                ai_ab = BenchmarkPlayer("X", fixed_depth=depth)
                move_alphabeta = ai_ab.getMoveAlphaBeta(test_board.copy())
                
                moves_total += 1
                if move_minmax == move_alphabeta:
                    moves_agree += 1
                
                # Add a random piece to change board state for next iteration
                if test_board.colFills[0] < test_board.numRows:
                    test_board.addPiece(0, "O")
            
            agreement_rate = moves_agree / moves_total if moves_total > 0 else 0
            
            quality_results.append({
                'board_size': f"{rows}x{cols}",
                'rows': rows,
                'cols': cols,
                'move_agreement_rate': agreement_rate,
                'moves_compared': moves_total
            })
            
            print(f"  Move agreement rate: {agreement_rate*100:.1f}%")
        
        return quality_results
    
    def measure_depth_reachability(self, board_sizes, time_limit=1.0):
        """
        Measure maximum depth reachable within time limit
        
        Returns: list of dicts with depth metrics
        """
        print("\n" + "="*60)
        print("Metric 6: Depth Reachability")
        print("="*60)
        
        depth_results = []
        
        for rows, cols in board_sizes:
            print(f"\nTesting board size: {rows}x{cols} (time limit: {time_limit}s)")
            game_board = board.Board(rows, cols, min(4, min(rows, cols)))
            
            # Test MinMax
            max_depth_minmax = self._find_max_depth_in_time(
                rows, cols, algorithm='minmax', time_limit=time_limit
            )
            
            # Test Alpha-Beta
            max_depth_ab = self._find_max_depth_in_time(
                rows, cols, algorithm='alphabeta', time_limit=time_limit
            )
            
            depth_improvement = max_depth_ab - max_depth_minmax
            
            depth_results.append({
                'board_size': f"{rows}x{cols}",
                'rows': rows,
                'cols': cols,
                'max_depth_minmax': max_depth_minmax,
                'max_depth_alphabeta': max_depth_ab,
                'depth_improvement': depth_improvement
            })
            
            print(f"  Max depth (MinMax): {max_depth_minmax}")
            print(f"  Max depth (Alpha-Beta): {max_depth_ab}")
            print(f"  Improvement: +{depth_improvement} levels")
        
        return depth_results
    
    def _find_max_depth_in_time(self, rows, cols, algorithm, time_limit):
        """Find maximum depth reachable within time limit"""
        game_board = board.Board(rows, cols, min(4, min(rows, cols)))
        depth = 1
        max_test_depth = 8  # Safety limit
        
        while depth <= max_test_depth:
            ai = BenchmarkPlayer("X", fixed_depth=depth)
            start_time = time.time()
            
            try:
                if algorithm == 'minmax':
                    ai.getMove(game_board.copy())
                else:
                    ai.getMoveAlphaBeta(game_board.copy())
                
                elapsed = time.time() - start_time
                
                if elapsed > time_limit:
                    return depth - 1
                
                depth += 1
            except (RecursionError, MemoryError) as e:
                return depth - 1
        
        return depth - 1
    
    def measure_memory_usage(self, board_sizes, depth=4):
        """
        Measure memory usage and recursion depth
        
        Returns: list of dicts with memory metrics
        """
        print("\n" + "="*60)
        print("Metric 7: Memory Efficiency")
        print("="*60)
        
        memory_results = []
        
        for rows, cols in board_sizes:
            print(f"\nTesting board size: {rows}x{cols} (depth={depth})")
            game_board = board.Board(rows, cols, min(4, min(rows, cols)))
            
            # Test MinMax
            tracemalloc.start()
            ai_minmax = BenchmarkPlayer("X", fixed_depth=depth)
            ai_minmax.getMove(game_board.copy())
            _, peak_minmax = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Test Alpha-Beta
            tracemalloc.start()
            ai_ab = BenchmarkPlayer("X", fixed_depth=depth)
            ai_ab.getMoveAlphaBeta(game_board.copy())
            _, peak_ab = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            memory_saving = 100 * (1 - (peak_ab / peak_minmax)) if peak_minmax > 0 else 0
            
            memory_results.append({
                'board_size': f"{rows}x{cols}",
                'rows': rows,
                'cols': cols,
                'peak_memory_minmax_kb': peak_minmax / 1024,
                'peak_memory_alphabeta_kb': peak_ab / 1024,
                'memory_saving_percent': memory_saving,
                'recursion_depth': depth
            })
            
            print(f"  Peak memory (MinMax): {peak_minmax/1024:.2f} KB")
            print(f"  Peak memory (Alpha-Beta): {peak_ab/1024:.2f} KB")
            print(f"  Memory saving: {memory_saving:.1f}%")
        
        return memory_results
    
    def run_comprehensive_benchmark(self, board_configs, depths=[3, 4], num_trials=1):
        """
        Run comprehensive benchmark with all metrics
        
        Args:
            board_configs: List of (rows, cols, winNum) tuples or (rows, cols) tuples
                          If winNum is not provided, defaults to min(4, min(rows, cols))
            depths: List of depths to test
            num_trials: Number of trials to run for statistical analysis (default: 1)
                       Note: For deterministic minimax on same board, results are identical
        
        Returns: DataFrame with all metrics (includes mean/std if num_trials > 1)
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE CONNECT 4 ALGORITHM BENCHMARK")
        print("="*80)
        
        all_metrics = []
        
        for config in board_configs:
            # Handle both (rows, cols) and (rows, cols, winNum) formats
            if len(config) == 3:
                rows, cols, winNum = config
            else:
                rows, cols = config
                winNum = min(4, min(rows, cols))  # Default behavior
            
            print(f"\n{'='*80}")
            print(f"Testing board size: {rows} rows x {cols} columns (winNum={winNum})")
            print(f"{'='*80}")
            
            for depth in depths:
                print(f"\n--- Search Depth: {depth} ---")
                if num_trials > 1:
                    print(f"  Running {num_trials} trials for statistical analysis...")
                
                # Collect metrics across trials
                trial_metrics = []
                
                for trial in range(num_trials):
                    game_board = board.Board(rows, cols, winNum)
                    
                    # Create AI instances
                    ai_minmax = BenchmarkPlayer("X", fixed_depth=depth)
                    ai_alphabeta = BenchmarkPlayer("X", fixed_depth=depth)
                    
                    # Measure MinMax
                    start_time = time.time()
                    move_minmax = ai_minmax.getMove(game_board.copy())
                    time_minmax = time.time() - start_time
                    
                    # Measure Alpha-Beta
                    start_time = time.time()
                    move_alphabeta = ai_alphabeta.getMoveAlphaBeta(game_board.copy())
                    time_alphabeta = time.time() - start_time
                    
                    # Calculate all metrics for this trial
                    nodes_minmax = ai_minmax.numExpanded
                    nodes_alphabeta = ai_alphabeta.numExpanded
                    pruning_count = ai_alphabeta.numPruned
                    
                    node_reduction = 1 - (nodes_alphabeta / nodes_minmax) if nodes_minmax > 0 else 0
                    time_speedup = time_minmax / time_alphabeta if time_alphabeta > 0 else 0
                    pruning_rate = pruning_count / (nodes_alphabeta + pruning_count) if (nodes_alphabeta + pruning_count) > 0 else 0
                    
                    # Effective branching factor
                    effective_bf_minmax = nodes_minmax ** (1.0/depth) if nodes_minmax > 0 and depth > 0 else 0
                    effective_bf_ab = nodes_alphabeta ** (1.0/depth) if nodes_alphabeta > 0 and depth > 0 else 0
                    
                    trial_metrics.append({
                        'nodes_minmax': nodes_minmax,
                        'nodes_alphabeta': nodes_alphabeta,
                        'node_reduction_percent': node_reduction * 100,
                        'time_minmax': time_minmax,
                        'time_alphabeta': time_alphabeta,
                        'time_speedup': time_speedup,
                        'pruning_count': pruning_count,
                        'pruning_rate': pruning_rate * 100,
                        'nodes_per_second_minmax': nodes_minmax / time_minmax if time_minmax > 0 else 0,
                        'nodes_per_second_alphabeta': nodes_alphabeta / time_alphabeta if time_alphabeta > 0 else 0,
                        'effective_bf_minmax': effective_bf_minmax,
                        'effective_bf_alphabeta': effective_bf_ab,
                        'branching_reduction': 1 - (effective_bf_ab / effective_bf_minmax) if effective_bf_minmax > 0 else 0,
                        'complexity_ratio': nodes_alphabeta / nodes_minmax if nodes_minmax > 0 else 0
                    })
                
                # Calculate statistics if multiple trials
                if num_trials > 1:
                    trial_df = pd.DataFrame(trial_metrics)
                    metrics = {
                        'rows': rows,
                        'cols': cols,
                        'winNum': winNum,
                        'board_size': f"{rows}x{cols}",
                        'board_config': f"{rows}x{cols} (winNum={winNum})",
                        'total_cells': rows * cols,
                        'theoretical_branching': cols,
                        'depth': depth,
                        'num_trials': num_trials,
                        
                        # Node metrics (mean)
                        'nodes_minmax': trial_df['nodes_minmax'].mean(),
                        'nodes_alphabeta': trial_df['nodes_alphabeta'].mean(),
                        'nodes_minmax_std': trial_df['nodes_minmax'].std(),
                        'nodes_alphabeta_std': trial_df['nodes_alphabeta'].std(),
                        'node_reduction_percent': trial_df['node_reduction_percent'].mean(),
                        'node_reduction_percent_std': trial_df['node_reduction_percent'].std(),
                        
                        # Time metrics (mean)
                        'time_minmax': trial_df['time_minmax'].mean(),
                        'time_alphabeta': trial_df['time_alphabeta'].mean(),
                        'time_minmax_std': trial_df['time_minmax'].std(),
                        'time_alphabeta_std': trial_df['time_alphabeta'].std(),
                        'time_speedup': trial_df['time_speedup'].mean(),
                        'time_speedup_std': trial_df['time_speedup'].std(),
                        
                        # Pruning metrics (mean)
                        'pruning_count': trial_df['pruning_count'].mean(),
                        'pruning_count_std': trial_df['pruning_count'].std(),
                        'pruning_rate': trial_df['pruning_rate'].mean(),
                        'pruning_rate_std': trial_df['pruning_rate'].std(),
                        
                        # Efficiency metrics (mean)
                        'nodes_per_second_minmax': trial_df['nodes_per_second_minmax'].mean(),
                        'nodes_per_second_alphabeta': trial_df['nodes_per_second_alphabeta'].mean(),
                        'nodes_per_second_minmax_std': trial_df['nodes_per_second_minmax'].std(),
                        'nodes_per_second_alphabeta_std': trial_df['nodes_per_second_alphabeta'].std(),
                        
                        # Branching factor (mean)
                        'effective_bf_minmax': trial_df['effective_bf_minmax'].mean(),
                        'effective_bf_alphabeta': trial_df['effective_bf_alphabeta'].mean(),
                        'effective_bf_minmax_std': trial_df['effective_bf_minmax'].std(),
                        'effective_bf_alphabeta_std': trial_df['effective_bf_alphabeta'].std(),
                        'branching_reduction': trial_df['branching_reduction'].mean(),
                        'branching_reduction_std': trial_df['branching_reduction'].std(),
                        
                        # Complexity (mean)
                        'complexity_ratio': trial_df['complexity_ratio'].mean(),
                        'complexity_ratio_std': trial_df['complexity_ratio'].std()
                    }
                else:
                    # Single trial (original behavior)
                    trial = trial_metrics[0]
                    metrics = {
                        'rows': rows,
                        'cols': cols,
                        'winNum': winNum,
                        'board_size': f"{rows}x{cols}",
                        'board_config': f"{rows}x{cols} (winNum={winNum})",
                        'total_cells': rows * cols,
                        'theoretical_branching': cols,
                        'depth': depth,
                        'num_trials': 1,
                        
                        # Node metrics
                        'nodes_minmax': trial['nodes_minmax'],
                        'nodes_alphabeta': trial['nodes_alphabeta'],
                        'node_reduction_percent': trial['node_reduction_percent'],
                        
                        # Time metrics
                        'time_minmax': trial['time_minmax'],
                        'time_alphabeta': trial['time_alphabeta'],
                        'time_speedup': trial['time_speedup'],
                        
                        # Pruning metrics
                        'pruning_count': trial['pruning_count'],
                        'pruning_rate': trial['pruning_rate'],
                        
                        # Efficiency metrics
                        'nodes_per_second_minmax': trial['nodes_per_second_minmax'],
                        'nodes_per_second_alphabeta': trial['nodes_per_second_alphabeta'],
                        
                        # Branching factor
                        'effective_bf_minmax': trial['effective_bf_minmax'],
                        'effective_bf_alphabeta': trial['effective_bf_alphabeta'],
                        'branching_reduction': trial['branching_reduction'],
                        
                        # Complexity
                        'complexity_ratio': trial['complexity_ratio']
                    }
                
                all_metrics.append(metrics)
                
                # Print summary
                self._print_metrics_summary(metrics)
        
        return pd.DataFrame(all_metrics)
    
    def _print_metrics_summary(self, metrics):
        """Print summary of metrics"""
        num_trials = metrics.get('num_trials', 1)
        has_stats = num_trials > 1
        
        print(f"  Board: {metrics['board_size']}, Depth: {metrics['depth']}, winNum: {metrics['winNum']}")
        
        if has_stats:
            print(f"  Nodes: MinMax={metrics['nodes_minmax']:.0f}±{metrics.get('nodes_minmax_std', 0):.0f}, "
                  f"AlphaBeta={metrics['nodes_alphabeta']:.0f}±{metrics.get('nodes_alphabeta_std', 0):.0f} "
                  f"(reduction: {metrics['node_reduction_percent']:.1f}±{metrics.get('node_reduction_percent_std', 0):.1f}%)")
            print(f"  Time: MinMax={metrics['time_minmax']:.4f}±{metrics.get('time_minmax_std', 0):.4f}s, "
                  f"AlphaBeta={metrics['time_alphabeta']:.4f}±{metrics.get('time_alphabeta_std', 0):.4f}s "
                  f"(speedup: {metrics['time_speedup']:.2f}±{metrics.get('time_speedup_std', 0):.2f}x)")
            print(f"  Pruning: {metrics['pruning_count']:.0f}±{metrics.get('pruning_count_std', 0):.0f} branches "
                  f"({metrics['pruning_rate']:.1f}±{metrics.get('pruning_rate_std', 0):.1f}% rate)")
        else:
            print(f"  Nodes: MinMax={metrics['nodes_minmax']:,}, "
                  f"AlphaBeta={metrics['nodes_alphabeta']:,} "
                  f"(reduction: {metrics['node_reduction_percent']:.1f}%)")
            print(f"  Time: MinMax={metrics['time_minmax']:.4f}s, "
                  f"AlphaBeta={metrics['time_alphabeta']:.4f}s "
                  f"(speedup: {metrics['time_speedup']:.2f}x)")
            print(f"  Pruning: {metrics['pruning_count']:,} branches "
                  f"({metrics['pruning_rate']:.1f}% rate)")
        
        print(f"  Effective BF: MinMax={metrics['effective_bf_minmax']:.2f}, "
              f"AlphaBeta={metrics['effective_bf_alphabeta']:.2f}")
    
    def visualize_results(self, df):
        """Create comprehensive visualizations of benchmark results"""
        print("\n" + "="*60)
        print("Generating Visualizations")
        print("="*60)
        
        if df.empty:
            print("No data to visualize!")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Connect 4 Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Nodes Visited Comparison
        ax1 = axes[0, 0]
        board_labels = df['board_size'].unique()
        x = np.arange(len(board_labels))
        width = 0.35
        
        # Group by board size and depth, take average
        df_grouped = df.groupby('board_size').agg({
            'nodes_minmax': 'mean',
            'nodes_alphabeta': 'mean'
        }).reset_index()
        
        ax1.bar(x - width/2, df_grouped['nodes_minmax'], width, 
                label='MinMax', color='#ff9933', alpha=0.8)
        ax1.bar(x + width/2, df_grouped['nodes_alphabeta'], width, 
                label='Alpha-Beta', color='#009999', alpha=0.8)
        ax1.set_xlabel('Board Size')
        ax1.set_ylabel('Nodes Visited')
        ax1.set_title('1. Nodes Visited Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_grouped['board_size'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Search Time Comparison
        ax2 = axes[0, 1]
        df_time = df.groupby('board_size').agg({
            'time_minmax': 'mean',
            'time_alphabeta': 'mean',
            'total_cells': 'first'
        }).reset_index()
        
        ax2.plot(df_time['total_cells'], df_time['time_minmax'], 
                'o-', label='MinMax', color='#ff9933', linewidth=2, markersize=8)
        ax2.plot(df_time['total_cells'], df_time['time_alphabeta'], 
                's-', label='Alpha-Beta', color='#009999', linewidth=2, markersize=8)
        ax2.set_xlabel('Board Size (Total Cells)')
        ax2.set_ylabel('Search Time (seconds)')
        ax2.set_title('2. Search Time vs Board Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. Pruning Efficiency
        ax3 = axes[1, 0]
        df_pruning = df.groupby('board_size').agg({
            'pruning_rate': 'mean',
            'pruning_count': 'mean'
        }).reset_index()
        
        bars = ax3.bar(range(len(df_pruning)), df_pruning['pruning_rate'], 
                      color='#009999', alpha=0.8)
        ax3.set_xlabel('Board Configuration')
        ax3.set_ylabel('Pruning Rate (%)')
        ax3.set_title('3. Alpha-Beta Pruning Efficiency')
        ax3.set_xticks(range(len(df_pruning)))
        ax3.set_xticklabels(df_pruning['board_size'], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, df_pruning['pruning_rate'])):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 4. Speedup Ratio
        ax4 = axes[1, 1]
        df_speedup = df.groupby('board_size').agg({
            'time_speedup': 'mean',
            'theoretical_branching': 'first'
        }).reset_index()
        
        ax4.plot(df_speedup['theoretical_branching'], df_speedup['time_speedup'], 
                'o-', color='#ff9933', linewidth=2, markersize=8)
        ax4.set_xlabel('Theoretical Branching Factor (Columns)')
        ax4.set_ylabel('Time Speedup (x)')
        ax4.set_title('4. Speedup vs Branching Factor')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        # 5. Complexity Reduction
        ax5 = axes[2, 0]
        df_complexity = df.groupby('board_size').agg({
            'complexity_ratio': 'mean',
            'total_cells': 'first'
        }).reset_index()
        
        ax5.plot(df_complexity['total_cells'], df_complexity['complexity_ratio'], 
                'o-', color='red', linewidth=2, markersize=8)
        ax5.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No improvement')
        ax5.set_xlabel('Board Size (Total Cells)')
        ax5.set_ylabel('Complexity Ratio (AlphaBeta/MinMax)')
        ax5.set_title('5. Complexity Reduction')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Node Processing Speed
        ax6 = axes[2, 1]
        df_speed = df.groupby('board_size').agg({
            'nodes_per_second_minmax': 'mean',
            'nodes_per_second_alphabeta': 'mean'
        }).reset_index()
        
        x_speed = np.arange(len(df_speed))
        ax6.bar(x_speed - width/2, df_speed['nodes_per_second_minmax'], width, 
               label='MinMax', color='#ff9933', alpha=0.8)
        ax6.bar(x_speed + width/2, df_speed['nodes_per_second_alphabeta'], width, 
               label='Alpha-Beta', color='#009999', alpha=0.8)
        ax6.set_xlabel('Board Size')
        ax6.set_ylabel('Nodes per Second')
        ax6.set_title('6. Node Processing Speed')
        ax6.set_xticks(x_speed)
        ax6.set_xticklabels(df_speed['board_size'], rotation=45, ha='right')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_yscale('log')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "performance_comparison.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved visualization to: {output_path}")
        
        # Also save as PNG
        output_path_png = self.output_dir / "performance_comparison.png"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        print(f"  Saved visualization to: {output_path_png}")
        
        plt.close()
        
        # Print summary statistics
        self._print_summary_statistics(df)
    
    def _print_summary_statistics(self, df):
        """Print summary statistics table"""
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*80)
        
        summary_df = df.groupby(['rows', 'cols']).agg({
            'node_reduction_percent': 'mean',
            'time_speedup': 'mean',
            'pruning_rate': 'mean',
            'complexity_ratio': 'mean',
            'branching_reduction': 'mean'
        }).round(2)
        
        print("\nAverage Metrics by Board Size:")
        print(summary_df.to_string())
        
        print("\nOverall Averages:")
        print(f"  Node Reduction: {df['node_reduction_percent'].mean():.1f}%")
        print(f"  Time Speedup: {df['time_speedup'].mean():.2f}x")
        print(f"  Pruning Rate: {df['pruning_rate'].mean():.1f}%")
        print(f"  Complexity Ratio: {df['complexity_ratio'].mean():.3f}")
        print(f"  Branching Reduction: {df['branching_reduction'].mean()*100:.1f}%")
    
    def analyze_winnum_impact(self, df):
        """
        Analyze the impact of winNum (number of pieces needed to win) on algorithm performance
        
        Args:
            df: DataFrame from run_comprehensive_benchmark
        
        Returns: DataFrame with winNum analysis
        """
        print("\n" + "="*80)
        print("WINNUM IMPACT ANALYSIS")
        print("="*80)
        
        if 'winNum' not in df.columns:
            print("  No winNum data available for analysis")
            return pd.DataFrame()
        
        # Group by board size and winNum
        winnum_analysis = df.groupby(['board_size', 'winNum', 'depth']).agg({
            'nodes_minmax': 'mean',
            'nodes_alphabeta': 'mean',
            'node_reduction_percent': 'mean',
            'time_minmax': 'mean',
            'time_alphabeta': 'mean',
            'time_speedup': 'mean',
            'pruning_rate': 'mean',
            'effective_bf_minmax': 'mean',
            'effective_bf_alphabeta': 'mean'
        }).reset_index()
        
        # Print summary
        print("\nImpact of winNum on Performance:")
        print("-" * 80)
        
        for board_size in winnum_analysis['board_size'].unique():
            board_data = winnum_analysis[winnum_analysis['board_size'] == board_size]
            print(f"\n{board_size} board:")
            for _, row in board_data.iterrows():
                print(f"  winNum={row['winNum']}, depth={row['depth']}: "
                      f"Nodes: {row['nodes_minmax']:.0f}→{row['nodes_alphabeta']:.0f} "
                      f"({row['node_reduction_percent']:.1f}% reduction), "
                      f"Speedup: {row['time_speedup']:.1f}x")
        
        # Calculate winNum effect (difference between min and max winNum for same board)
        winnum_effect = []
        for board_size in winnum_analysis['board_size'].unique():
            board_data = winnum_analysis[winnum_analysis['board_size'] == board_size]
            for depth in board_data['depth'].unique():
                depth_data = board_data[board_data['depth'] == depth]
                if len(depth_data) > 1:
                    min_winnum = depth_data['winNum'].min()
                    max_winnum = depth_data['winNum'].max()
                    min_data = depth_data[depth_data['winNum'] == min_winnum].iloc[0]
                    max_data = depth_data[depth_data['winNum'] == max_winnum].iloc[0]
                    
                    winnum_effect.append({
                        'board_size': board_size,
                        'depth': depth,
                        'winNum_range': f"{min_winnum}-{max_winnum}",
                        'nodes_minmax_diff': max_data['nodes_minmax'] - min_data['nodes_minmax'],
                        'nodes_alphabeta_diff': max_data['nodes_alphabeta'] - min_data['nodes_alphabeta'],
                        'reduction_diff': max_data['node_reduction_percent'] - min_data['node_reduction_percent'],
                        'speedup_diff': max_data['time_speedup'] - min_data['time_speedup']
                    })
        
        if winnum_effect:
            effect_df = pd.DataFrame(winnum_effect)
            print("\n" + "="*80)
            print("winNum Effect Summary (difference between min and max winNum):")
            print("-" * 80)
            print(effect_df.to_string(index=False))
            return winnum_analysis, effect_df
        
        return winnum_analysis, pd.DataFrame()
    
    def verify_report_data(self, df, report_table_data=None):
        """
        Verify that benchmark data matches report table values
        
        Args:
            df: DataFrame from run_comprehensive_benchmark
            report_table_data: Optional dict with expected values from report
                             Format: {(board, depth): {'minmax': int, 'alphabeta': int}}
                             If None, will only verify depths that exist in the benchmark data
        
        Returns: DataFrame with verification results
        """
        print("\n" + "="*80)
        print("DATA VERIFICATION: Benchmark vs Report Tables")
        print("="*80)
        
        # Default report table data from report.tex
        if report_table_data is None:
            report_table_data = {
                ('4x4', 3): {'minmax': 340, 'alphabeta': 61, 'reduction': 82.1},
                ('4x4', 4): {'minmax': 1360, 'alphabeta': 139, 'reduction': 89.8},
                ('5x5', 3): {'minmax': 780, 'alphabeta': 92, 'reduction': 88.2},
                ('5x5', 4): {'minmax': 3905, 'alphabeta': 241, 'reduction': 93.8},
                ('6x7', 3): {'minmax': 2800, 'alphabeta': 172, 'reduction': 93.9},
                ('6x7', 4): {'minmax': 19607, 'alphabeta': 563, 'reduction': 97.1},
                ('8x8', 3): {'minmax': 4680, 'alphabeta': 221, 'reduction': 95.3},
                ('8x8', 4): {'minmax': 37448, 'alphabeta': 796, 'reduction': 97.9}
            }
        
        # Filter to only check depths that exist in the benchmark data
        available_depths = df['depth'].unique() if not df.empty else []
        filtered_report_data = {
            (board, depth): expected 
            for (board, depth), expected in report_table_data.items()
            if depth in available_depths
        }
        
        if len(filtered_report_data) < len(report_table_data):
            skipped = len(report_table_data) - len(filtered_report_data)
            print(f"\nNote: Skipping {skipped} verification(s) for depths not tested in this benchmark.")
            print(f"Available depths in benchmark: {sorted(available_depths)}")
            print(f"Verifying only depths: {sorted(set(d for _, d in filtered_report_data.keys()))}\n")
        
        verification_results = []
        
        for (board_size, depth), expected in filtered_report_data.items():
            # Find matching row in benchmark data
            # Try to match by board_size and depth, prefer winNum=4 (matches report tables)
            matches = df[(df['board_size'] == board_size) & (df['depth'] == depth)]
            
            if len(matches) == 0:
                verification_results.append({
                    'board_size': board_size,
                    'depth': depth,
                    'winNum': 'N/A',
                    'status': 'NOT FOUND',
                    'expected_minmax': expected['minmax'],
                    'expected_alphabeta': expected['alphabeta'],
                    'expected_reduction': expected['reduction'],
                    'actual_minmax': None,
                    'actual_alphabeta': None,
                    'actual_reduction': None,
                    'minmax_diff': None,
                    'alphabeta_diff': None,
                    'reduction_diff': None
                })
                continue
            
            # Prefer winNum=4 if available (matches report tables), otherwise use first match
            winNum4_matches = matches[matches['winNum'] == 4]
            if len(winNum4_matches) > 0:
                actual = winNum4_matches.iloc[0]
            else:
                actual = matches.iloc[0]
            
            actual_minmax = int(actual['nodes_minmax'])
            actual_alphabeta = int(actual['nodes_alphabeta'])
            actual_reduction = actual['node_reduction_percent']
            actual_winNum = actual.get('winNum', 'N/A')
            
            minmax_diff = actual_minmax - expected['minmax']
            alphabeta_diff = actual_alphabeta - expected['alphabeta']
            reduction_diff = abs(actual_reduction - expected['reduction'])
            
            # Consider match if within 1% tolerance for reduction, exact match for nodes
            status = 'MATCH'
            if abs(minmax_diff) > 0 or abs(alphabeta_diff) > 0:
                status = 'NODE MISMATCH'
            if reduction_diff > 0.5:  # 0.5% tolerance
                status = 'REDUCTION MISMATCH' if status == 'MATCH' else 'MISMATCH'
            
            verification_results.append({
                'board_size': board_size,
                'depth': depth,
                'winNum': actual_winNum if pd.notna(actual_winNum) else 'N/A',
                'status': status,
                'expected_minmax': expected['minmax'],
                'expected_alphabeta': expected['alphabeta'],
                'expected_reduction': expected['reduction'],
                'actual_minmax': actual_minmax,
                'actual_alphabeta': actual_alphabeta,
                'actual_reduction': round(actual_reduction, 1),
                'minmax_diff': float(minmax_diff),
                'alphabeta_diff': float(alphabeta_diff),
                'reduction_diff': round(reduction_diff, 1)
            })
        
        verification_df = pd.DataFrame(verification_results)
        
        # Print results
        print("\nVerification Results:")
        print("-" * 80)
        for _, row in verification_df.iterrows():
            status_symbol = "✓" if row['status'] == 'MATCH' else "✗"
            winNum_str = f"winNum={row['winNum']}" if pd.notna(row['winNum']) and row['winNum'] != 'N/A' else "winNum=N/A"
            print(f"{status_symbol} {row['board_size']} depth {row['depth']} ({winNum_str}): {row['status']}")
            if row['status'] != 'MATCH':
                print(f"    Expected: {row['expected_minmax']}→{row['expected_alphabeta']} ({row['expected_reduction']}%)")
                print(f"    Actual:   {row['actual_minmax']}→{row['actual_alphabeta']} ({row['actual_reduction']}%)")
                if pd.notna(row['minmax_diff']) and row['minmax_diff'] != 0:
                    print(f"    MinMax diff: {row['minmax_diff']:+.0f}")
                if pd.notna(row['alphabeta_diff']) and row['alphabeta_diff'] != 0:
                    print(f"    AlphaBeta diff: {row['alphabeta_diff']:+.0f}")
                if pd.notna(row['reduction_diff']) and row['reduction_diff'] > 0.1:
                    print(f"    Reduction diff: {row['reduction_diff']:.1f}%")
        
        matches = len(verification_df[verification_df['status'] == 'MATCH'])
        total = len(verification_df)
        print(f"\nSummary: {matches}/{total} configurations match report data")
        
        return verification_df
    
    def save_results(self, df, filename='benchmark_results.csv'):
        """Save results to CSV file"""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        return output_path


def main():
    """Main function to run benchmarks"""
    
    # Create benchmark instance
    benchmark = ConnectFourBenchmark()
    
    # Define board configurations to test
    board_configs = [
        (4, 4),    # Small board
        (5, 5),    # Medium board
        (6, 7),    # Standard Connect 4
        (8, 8),    # Large board
    ]
    
    # Run comprehensive benchmark
    print("\nStarting comprehensive benchmark...")
    results_df = benchmark.run_comprehensive_benchmark(
        board_configs, 
        depths=[3, 4]  # Test different depths
    )
    
    # Visualize results
    benchmark.visualize_results(results_df)
    
    # Save results
    benchmark.save_results(results_df)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {benchmark.output_dir}")
    print("  - benchmark_results.csv: Detailed metrics")
    print("  - performance_comparison.pdf: Visualization")
    print("  - performance_comparison.png: Visualization")


if __name__ == "__main__":
    main()
