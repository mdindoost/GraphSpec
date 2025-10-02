"""
Compare all eigenspace transformation strategies.

This script tests all available eigenspace scaling strategies and compares
their performance on node classification.
"""

import sys
sys.path.append('..')

import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from sklearn.preprocessing import StandardScaler
from src.transformations.eigenspace import EigenspaceTransformation
from src.data.graph_utils import compute_normalized_laplacian
from src.training.trainer import Trainer
import argparse
import json
from pathlib import Path


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
    dataset = Planetoid(root='../data/raw', name=dataset_name, split='public')
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
    
    # Modify trainer to use train+val
    trainer = Trainer(
        input_dim=data.num_features,
        hidden_dim=hidden_dim,
        output_dim=dataset.num_classes,
        model_type='mlp',
        device=device
    )
    
    # Quick hack: replace train_mask with train_val_mask
    original_train_mask = data_raw.train_mask.clone()
    data_raw.train_mask = train_val_mask
    
    result_raw = trainer.train(data_raw, epochs=epochs, verbose=False)
    baseline_acc = result_raw['test_acc']
    
    # Restore
    data_raw.train_mask = original_train_mask
    
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
            data_transformed.train_mask = train_val_mask
            
            trainer = Trainer(
                input_dim=X_transformed.shape[1],
                hidden_dim=hidden_dim,
                output_dim=dataset.num_classes,
                model_type='mlp',
                device=device
            )
            
            result = trainer.train(data_transformed, epochs=epochs, verbose=False)
            
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
    output_dir = Path('../results/metrics')
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
