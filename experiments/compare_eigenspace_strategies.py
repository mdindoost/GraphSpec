"""
Compare all eigenspace transformation strategies.

This script tests all available eigenspace scaling strategies and compares
their performance on node classification.
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


def compare_strategies(dataset_name='Cora', hidden_dim=64, epochs=500, device='cpu'):
    """
    Compare all eigenspace transformation strategies.
    
    Args:
        dataset_name: Dataset to use
        hidden_dim: Hidden dimension for MLP
        epochs: Number of training epochs
        device: Device to use
    """
    print("\n" + "="*70)
    print(f"COMPARING EIGENSPACE STRATEGIES - {dataset_name}")
    print("="*70)
    
    # Load dataset
    dataset = Planetoid(root='data/raw', name=dataset_name, split='public')
    data = dataset[0]
    
    # Use train+val for training (standard for public split)
    train_val_mask = data.train_mask | data.val_mask
    
    print(f"\nDataset: {dataset_name}")
    print(f"  Train+Val: {train_val_mask.sum()} samples")
    print(f"  Test: {data.test_mask.sum()} samples")
    
    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(data.x.numpy())
    
    # Compute Laplacian
    edge_index_np = data.edge_index.numpy()
    L_norm = compute_normalized_laplacian(edge_index_np, data.num_nodes)
    
    # Get all strategies
    strategies = list(EigenspaceTransformation.STRATEGIES.keys())
    
    print(f"\nTesting {len(strategies)} strategies...")
    
    # Test raw MLP first (baseline)
    print("\n" + "-"*70)
    print("BASELINE: Raw MLP (no transformation)")
    print("-"*70)
    
    data_raw = data.clone()
    data_raw.x = torch.FloatTensor(X_normalized)

    # Create MLP with dropout=0.8
    model = MLP(
        input_dim=data.num_features,
        hidden_dim=hidden_dim,
        output_dim=dataset.num_classes,
        dropout=0.8
    ).to(device)

    # Create optimizer with lr=0.01, weight_decay=1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

    result_raw = train_model(model, data_raw, optimizer, train_val_mask, epochs=epochs, use_graph=False)
    baseline_acc = result_raw['test_acc']
    
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    
    # Test each strategy
    results = {}
    
    for i, strategy in enumerate(strategies, 1):
        print("\n" + "-"*70)
        print(f"[{i}/{len(strategies)}] Strategy: {strategy}")
        print(f"Description: {EigenspaceTransformation.STRATEGIES[strategy]}")
        print("-"*70)
        
        try:
            # Transform features
            transform = EigenspaceTransformation(target_dim=None, strategy=strategy)
            X_transformed = transform.fit_transform(X_normalized, L_norm)

            # Train MLP
            data_transformed = data.clone()
            data_transformed.x = torch.FloatTensor(X_transformed)

            # Create MLP with dropout=0.8
            model = MLP(
                input_dim=X_transformed.shape[1],
                hidden_dim=hidden_dim,
                output_dim=dataset.num_classes,
                dropout=0.8
            ).to(device)

            # Create optimizer with lr=0.01, weight_decay=1e-3
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

            result = train_model(model, data_transformed, optimizer, train_val_mask, epochs=epochs, use_graph=False)

            acc = result['test_acc']
            improvement = (acc - baseline_acc) * 100
            
            results[strategy] = {
                'accuracy': acc,
                'improvement': improvement,
                'eigenvalue_stats': transform.get_eigenvalue_stats()
            }
            
            symbol = "🏆" if improvement > 5 else "✅" if improvement > 2 else "➖" if improvement > 0 else "❌"
            print(f"Result: {acc:.4f} ({improvement:+.1f}% vs raw) {symbol}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            results[strategy] = {
                'accuracy': 0.0,
                'improvement': -100.0,
                'error': str(e)
            }
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - SORTED BY PERFORMANCE")
    print("="*70)
    
    print(f"\n{'Rank':<6} {'Strategy':<25} {'Accuracy':>10} {'vs Raw':>12}")
    print("-"*70)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for rank, (strategy, result) in enumerate(sorted_results, 1):
        acc = result['accuracy']
        imp = result['improvement']
        
        if imp > 5:
            symbol = "🏆"
        elif imp > 2:
            symbol = "✅"
        elif imp > 0:
            symbol = "➖"
        else:
            symbol = "❌"
        
        print(f"{rank:<6} {strategy:<25} {acc:>10.4f} {imp:>+11.1f}% {symbol}")
    
    # Save results
    output_dir = Path('results/metrics')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'eigenspace_strategies_{dataset_name}.json'
    
    save_data = {
        'dataset': dataset_name,
        'baseline_accuracy': baseline_acc,
        'strategies': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print winner
    winner_strategy, winner_result = sorted_results[0]
    print("\n" + "="*70)
    print("🏆 WINNER")
    print("="*70)
    print(f"Strategy: {winner_strategy}")
    print(f"Accuracy: {winner_result['accuracy']:.4f}")
    print(f"Improvement: {winner_result['improvement']:+.1f}%")
    print(f"Description: {EigenspaceTransformation.STRATEGIES[winner_strategy]}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', 
                       choices=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    # List available strategies first
    EigenspaceTransformation.list_strategies()
    
    # Run comparison
    compare_strategies(
        dataset_name=args.dataset,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        device=args.device
    )
