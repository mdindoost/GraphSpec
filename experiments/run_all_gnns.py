
"""
GNN comparison: Test multiple GNN architectures
"""

import sys
import os
from pathlib import Path

# Get project root and change working directory
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.append('.')

import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from src.transformations.eigenspace import EigenspaceTransformation
from src.data.graph_utils import compute_normalized_laplacian
from src.training.trainer import Trainer
import argparse
import json


def run_gnn_comparison(dataset_name='Cora', hidden_dim=64, epochs=200,
                       num_runs=10, device='cuda'):
    """
    Compare eigenspace+MLP against multiple GNN architectures.
    """
    
    print(f"\n{'='*80}")
    print(f"GNN COMPARISON: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Load dataset
    dataset = Planetoid(root='data/raw', name=dataset_name)
    data = dataset[0]
    
    # Prepare eigenspace transformation
    X_np = data.x.cpu().numpy()
    edge_index_np = data.edge_index.cpu().numpy()
    L_norm = compute_normalized_laplacian(edge_index_np, data.num_nodes)
    
    # GNN models to test
    gnn_models = ['gcn', 'gat', 'sage']
    
    # Storage
    results = {
        'eigenspace_mlp': [],
        **{model: [] for model in gnn_models}
    }
    
    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run+1}/{num_runs}")
        print(f"{'='*60}")
        
        # Eigenspace + MLP
        print("\n[1/4] Eigenspace + MLP...")
        eigen_transform = EigenspaceTransformation(target_dim=None)
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
        result = trainer.train(data_eigen, epochs=epochs, verbose=False)
        results['eigenspace_mlp'].append(result)
        
        # GNN models
        for i, gnn_type in enumerate(gnn_models):
            print(f"\n[{i+2}/4] {gnn_type.upper()}...")
            trainer = Trainer(
                input_dim=data.num_features,
                hidden_dim=hidden_dim,
                output_dim=dataset.num_classes,
                model_type=gnn_type,
                device=device
            )
            result = trainer.train(data, epochs=epochs, verbose=False)
            results[gnn_type].append(result)
    
    # Summary
    print(f"\n\n{'='*80}")
    print(f"GNN COMPARISON SUMMARY ({num_runs} runs)")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<20} {'Accuracy':>15} {'F1-Micro':>15} {'Time (s)':>12}")
    print(f"{'-'*80}")
    
    summary = {}
    for model_name, model_results in results.items():
        accs = [r['test_acc'] for r in model_results]
        f1s = [r['test_f1_micro'] for r in model_results]
        times = [r['train_time'] for r in model_results]
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_f1 = np.mean(f1s)
        mean_time = np.mean(times)
        
        print(f"{model_name:<20} {mean_acc:>7.4f}±{std_acc:.4f} "
              f"{mean_f1:>7.4f}±{np.std(f1s):.4f} {mean_time:>12.2f}")
        
        summary[model_name] = {
            'accuracy_mean': float(mean_acc),
            'accuracy_std': float(std_acc),
            'f1_mean': float(mean_f1),
            'time_mean': float(mean_time)
        }
    
    # Save results
    output_dir = Path('results/metrics')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'gnn_comparison_{dataset_name}.json'
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    run_gnn_comparison(
        dataset_name=args.dataset,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        num_runs=args.runs,
        device=args.device
    )
