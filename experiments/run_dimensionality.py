
"""
Dimensionality experiment: Test different target dimensions K
"""

import sys
sys.path.append('..')

import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from src.transformations.eigenspace import EigenspaceTransformation
from src.transformations.random import RandomTransformation
from src.data.graph_utils import compute_normalized_laplacian
from src.training.trainer import Trainer
import argparse
import json
from pathlib import Path


def run_dimensionality_experiment(dataset_name='Cora', hidden_dim=64, epochs=200,
                                  target_dims=None, num_runs=5, device='cuda'):
    """
    Test different target dimensions K.
    Compare K < D, K = D, K > D for both random and eigenspace.
    """
    
    print(f"\n{'='*80}")
    print(f"DIMENSIONALITY EXPERIMENT: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Load dataset
    dataset = Planetoid(root='../data/raw', name=dataset_name)
    data = dataset[0]
    original_dim = data.num_features
    
    if target_dims is None:
        # Test K < D, K = D, K > D
        target_dims = [
            original_dim // 4,  # K << D
            original_dim // 2,  # K < D
            original_dim,       # K = D
            original_dim * 2,   # K > D
            original_dim * 4    # K >> D
        ]
    
    print(f"Dataset: {dataset_name}")
    print(f"  Original dimension: {original_dim}")
    print(f"  Target dimensions: {target_dims}\n")
    
    # Prepare data
    X_np = data.x.cpu().numpy()
    edge_index_np = data.edge_index.cpu().numpy()
    L_norm = compute_normalized_laplacian(edge_index_np, data.num_nodes)
    
    # Storage
    results = {
        'random': {dim: [] for dim in target_dims},
        'eigenspace': {dim: [] for dim in target_dims}
    }
    
    for target_dim in target_dims:
        print(f"\n{'='*60}")
        print(f"Testing dimension: {target_dim} (ratio: {target_dim/original_dim:.2f})")
        print(f"{'='*60}")
        
        for run in range(num_runs):
            print(f"\nRun {run+1}/{num_runs}")
            
            # Random projection
            print(f"  [Random] K={target_dim}...")
            random_transform = RandomTransformation(target_dim=target_dim, seed=run)
            X_random = random_transform.fit_transform(X_np)
            data_random = data.clone()
            data_random.x = torch.FloatTensor(X_random)
            
            trainer = Trainer(
                input_dim=X_random.shape[1],
                hidden_dim=hidden_dim,
                output_dim=dataset.num_classes,
                model_type='mlp',
                device=device
            )
            result_random = trainer.train(data_random, epochs=epochs, verbose=False)
            results['random'][target_dim].append(result_random)
            
            # Eigenspace projection
            print(f"  [Eigenspace] K={target_dim}...")
            eigen_transform = EigenspaceTransformation(target_dim=target_dim)
            X_eigen = eigen_transform.fit_transform(X_np, L_norm)
            data_eigen = data.clone()
            data_eigen.x = torch.FloatTensor(X_eigen)
            
            trainer = Trainer(
                input_dim=X_eigen.shape[1],
                hidden_dim=hidden_dim,
                output_dim=dataset.num_classes,
                model_type='mlp',
                device=device
            )
            result_eigen = trainer.train(data_eigen, epochs=epochs, verbose=False)
            results['eigenspace'][target_dim].append(result_eigen)
    
    # Print summary
    print(f"\n\n{'='*80}")
    print(f"DIMENSIONALITY RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Dimension':<12} {'Ratio':<8} {'Random Acc':>15} {'Eigenspace Acc':>15} {'Improvement':>15}")
    print(f"{'-'*80}")
    
    summary = {}
    for target_dim in target_dims:
        random_accs = [r['test_acc'] for r in results['random'][target_dim]]
        eigen_accs = [r['test_acc'] for r in results['eigenspace'][target_dim]]
        
        random_mean = np.mean(random_accs)
        random_std = np.std(random_accs)
        eigen_mean = np.mean(eigen_accs)
        eigen_std = np.std(eigen_accs)
        improvement = eigen_mean - random_mean
        
        ratio = target_dim / original_dim
        
        print(f"{target_dim:<12} {ratio:<8.2f} {random_mean:>7.4f}±{random_std:.4f} "
              f"{eigen_mean:>7.4f}±{eigen_std:.4f} {improvement:>+7.4f}")
        
        summary[str(target_dim)] = {
            'random_mean': float(random_mean),
            'random_std': float(random_std),
            'eigenspace_mean': float(eigen_mean),
            'eigenspace_std': float(eigen_std),
            'improvement': float(improvement)
        }
    
    # Save results
    output_dir = Path('../results/metrics')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'dimensionality_{dataset_name}.json'
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dims', type=int, nargs='+', default=None,
                       help='Custom target dimensions')
    
    args = parser.parse_args()
    
    run_dimensionality_experiment(
        dataset_name=args.dataset,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        target_dims=args.dims,
        num_runs=args.runs,
        device=args.device
    )
