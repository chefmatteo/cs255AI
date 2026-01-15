"""
Simple script to run Connect 4 algorithm performance benchmarks
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark import ConnectFourBenchmark

def main():
    """Run benchmark with customizable parameters"""
    
    print("Connect 4 Algorithm Performance Benchmark")
    print("\nThis will compare MinMax vs Alpha-Beta algorithms")
    print("on different board sizes and search depths.\n")
    
    # Configuration: (rows, columns, winNum) tuples
    # winNum = number of pieces needed in a line to win
    # Note: Configurations matching report tables are included for verification
    board_configs = [
        # Report table configurations (for verification)
        (4, 4, 4),    # Small board, need 4 in a row (matches report)
        (5, 5, 4),    # Medium board, need 4 in a row (matches report - 5×5)
        (6, 7, 4),    # Standard Connect 4 (matches report)
        (8, 8, 4),    # Large board, need 4 in a row (matches report)
        
        # Additional configurations for winNum analysis
        (4, 4, 3),    # Small board, need 3 in a row
        (5, 6, 4),    # Medium board (5×6), need 4 in a row
        (5, 6, 3),    # Medium board (5×6), need 3 in a row
        (8, 8, 7),    # Large board, need 7 in a row
        (8, 8, 2),    # Large board, need 2 in a row (very easy to win)
    ]
    
    depths = [4, 5]  # Search depths to test
    
    print(f"Board configurations: {[f'{r}x{c} (winNum={w})' for r, c, w in board_configs]}")
    print(f"Search depths: {depths}")
    print("\nStarting benchmark...\n")
    
    # Create benchmark instance
    benchmark = ConnectFourBenchmark()
    
    # Run comprehensive benchmark
    # Set num_trials > 1 for statistical analysis (mean/std)
    # Note: For deterministic minimax on same board, results are identical
    num_trials = 1  # Change to 3-5 if you want statistical analysis
    
    results_df = benchmark.run_comprehensive_benchmark(
        board_configs, 
        depths=depths,
        num_trials=num_trials
    )
    
    # Verify data matches report tables
    print("\n" + "="*80)
    print("VERIFYING DATA AGAINST REPORT TABLES")
    print("="*80)
    verification_df = benchmark.verify_report_data(results_df)
    
    # Analyze winNum impact
    winnum_analysis, winnum_effect = benchmark.analyze_winnum_impact(results_df)
    
    # Visualize results
    print("\nGenerating visualizations...")
    benchmark.visualize_results(results_df)
    
    # Save results
    benchmark.save_results(results_df, 'benchmark_results.csv')
    
    # Save additional analyses
    if not verification_df.empty:
        benchmark.save_results(verification_df, 'data_verification.csv')
    if not winnum_analysis.empty:
        benchmark.save_results(winnum_analysis, 'winnum_analysis.csv')
    if not winnum_effect.empty:
        benchmark.save_results(winnum_effect, 'winnum_effect.csv')
    
    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)
    print(f"\nResults saved in: {benchmark.output_dir}")
    print("  - benchmark_results.csv: Detailed metrics")
    print("  - data_verification.csv: Verification against report tables")
    print("  - winnum_analysis.csv: winNum impact analysis")
    print("  - winnum_effect.csv: winNum effect summary")
    print("  - performance_comparison.pdf: Visualization")
    print("  - performance_comparison.png: Visualization")
    print("\nYou can now analyze the results and visualizations!")


if __name__ == "__main__":
    main()
