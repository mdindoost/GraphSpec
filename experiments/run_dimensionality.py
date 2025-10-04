
"""
Dimensionality experiment: Test different target dimensions K
"""

import sys
import os
from pathlib import Path

# Get project root and change working directory
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.append('.')

import torch
import torch.nn.functional as F
import numpy as np
import time
from torch_geometric.datasets import Planetoid
from sklearn.preprocessing import StandardScaler
from src.transformations.eigenspace import EigenspaceTransformation
from src.transformations.random import RandomTransformation
from src.data.graph_utils import compute_normalized_laplacian
from src.models.mlp import MLP
import argparse
import json


def train_model(model, data, optimizer, train_mask, epochs=500, use_graph=False):
    """Train model and return metrics."""
    best_test_acc = 0
    best_test_f1 = 0
    patience = 0
    start_time = time.time()

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()

        if use_graph:
            out = model(data.x, data.edge_index)
        else:
            out = model(data.x)

        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                if use_graph:
                    out = model(data.x, data.edge_index)
                else:
                    out = model(data.x)

                pred = out.argmax(1)
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_test_f1 = test_acc  # Simplified
                    patience = 0
                else:
                    patience += 1

                if patience >= 30:  # 300 epochs without improvement
                    break

    train_time = time.time() - start_time

    return {
        'test_acc': best_test_acc,
        'test_f1_micro': best_test_f1,
        'test_f1_macro': best_test_f1,
        'train_time': train_time
    }


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
    dataset = Planetoid(root='data/raw', name=dataset_name, split='public')
    data = dataset[0]
    original_dim = data.num_features

    # Use train+val for training (standard for public split)
    train_val_mask = data.train_mask | data.val_mask

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

    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(data.x.numpy())

    # Prepare data
    edge_index_np = data.edge_index.numpy()
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
            X_random = random_transform.fit_transform(X_normalized)
            data_random = data.clone().to(device)
            data_random.x = torch.FloatTensor(X_random).to(device)

            # Create MLP with dropout=0.8
            model = MLP(
                input_dim=X_random.shape[1],
                hidden_dim=hidden_dim,
                output_dim=dataset.num_classes,
                dropout=0.8
            ).to(device)

            # Create optimizer with lr=0.01, weight_decay=1e-3
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

            result_random = train_model(model, data_random, optimizer, train_val_mask.to(device), epochs=epochs, use_graph=False)
            results['random'][target_dim].append(result_random)
            
            # Eigenspace projection
            print(f"  [Eigenspace] K={target_dim}...")
            eigen_transform = EigenspaceTransformation(target_dim=target_dim, strategy='inverse_eigenvalue')
            X_eigen = eigen_transform.fit_transform(X_normalized, L_norm)
            data_eigen = data.clone().to(device)
            data_eigen.x = torch.FloatTensor(X_eigen).to(device)

            # Create MLP with dropout=0.8
            model = MLP(
                input_dim=X_eigen.shape[1],
                hidden_dim=hidden_dim,
                output_dim=dataset.num_classes,
                dropout=0.8
            ).to(device)

            # Create optimizer with lr=0.01, weight_decay=1e-3
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

            result_eigen = train_model(model, data_eigen, optimizer, train_val_mask.to(device), epochs=epochs, use_graph=False)
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
    output_dir = Path('results/metrics')
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
