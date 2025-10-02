
"""
Generate all plots from saved results.
"""

import sys
sys.path.append('..')

import json
from pathlib import Path
from src.utils.visualization import (
    plot_baseline_comparison,
    plot_dimensionality_results,
    plot_gnn_comparison
)


def generate_all_plots():
    """Generate all plots from saved results."""
    
    results_dir = Path('../results/metrics')
    plots_dir = Path('../results/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Baseline plots
    for dataset in ['Cora', 'CiteSeer', 'PubMed']:
        baseline_file = results_dir / f'baseline_{dataset}.json'
        if baseline_file.exists():
            with open(baseline_file) as f:
                results = json.load(f)
            
            save_path = plots_dir / f'baseline_{dataset}.png'
            plot_baseline_comparison(results, dataset, save_path)
    
    # Dimensionality plots
    for dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dim_file = results_dir / f'dimensionality_{dataset}.json'
        if dim_file.exists():
            with open(dim_file) as f:
                results = json.load(f)
            
            save_path = plots_dir / f'dimensionality_{dataset}.png'
            plot_dimensionality_results(results, dataset, save_path)
    
    # GNN comparison plots
    for dataset in ['Cora', 'CiteSeer', 'PubMed']:
        gnn_file = results_dir / f'gnn_comparison_{dataset}.json'
        if gnn_file.exists():
            with open(gnn_file) as f:
                results = json.load(f)
            
            save_path = plots_dir / f'gnn_comparison_{dataset}.png'
            plot_gnn_comparison(results, dataset, save_path)
    
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    generate_all_plots()

