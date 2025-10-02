
"""
Multi-dataset experiment: Run baseline on multiple datasets
"""

import sys
sys.path.append('..')

from run_baseline import run_baseline_experiment
import argparse
import json
from pathlib import Path


def run_all_datasets(datasets=None, hidden_dim=64, epochs=200, 
                     num_runs=10, device='cuda'):
    """Run baseline experiment on multiple datasets."""
    
    if datasets is None:
        datasets = ['Cora', 'CiteSeer', 'PubMed']
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n\n{'#'*80}")
        print(f"# DATASET: {dataset}")
        print(f"{'#'*80}\n")
        
        results = run_baseline_experiment(
            dataset_name=dataset,
            hidden_dim=hidden_dim,
            epochs=epochs,
            num_runs=num_runs,
            device=device
        )
        
        all_results[dataset] = results
    
    # Save combined results
    output_dir = Path('../results/metrics')
    output_file = output_dir / 'all_datasets_summary.json'
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nCombined results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    run_all_datasets(
        datasets=args.datasets,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        num_runs=args.runs,
        device=args.device
    )
