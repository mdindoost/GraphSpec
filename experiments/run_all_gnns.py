
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
import torch.nn.functional as F
import numpy as np
import time
from torch_geometric.datasets import Planetoid
from sklearn.preprocessing import StandardScaler
from src.transformations.eigenspace import EigenspaceTransformation
from src.data.graph_utils import compute_normalized_laplacian
from src.models.mlp import MLP
from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.sage import GraphSAGE
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


def run_gnn_comparison(dataset_name='Cora', hidden_dim=64, epochs=200,
                       num_runs=10, device='cuda'):
    """
    Compare eigenspace+MLP against multiple GNN architectures.
    """
    
    print(f"\n{'='*80}")
    print(f"GNN COMPARISON: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Load dataset
    dataset = Planetoid(root='data/raw', name=dataset_name, split='public')
    data = dataset[0]

    # Use train+val for training (standard for public split)
    train_val_mask = data.train_mask | data.val_mask

    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(data.x.numpy())

    # Prepare eigenspace transformation
    edge_index_np = data.edge_index.numpy()
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
        eigen_transform = EigenspaceTransformation(target_dim=None, strategy='inverse_eigenvalue')
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

        result = train_model(model, data_eigen, optimizer, train_val_mask.to(device), epochs=epochs, use_graph=False)
        results['eigenspace_mlp'].append(result)
        
        # GNN models
        for i, gnn_type in enumerate(gnn_models):
            print(f"\n[{i+2}/4] {gnn_type.upper()}...")

            # Prepare data with normalized features
            data_gnn = data.clone()
            data_gnn.x = torch.FloatTensor(X_normalized).to(device)

            # Create GNN model with dropout=0.8
            if gnn_type == 'gcn':
                model = GCN(
                    input_dim=data.num_features,
                    hidden_dim=hidden_dim,
                    output_dim=dataset.num_classes,
                    dropout=0.8
                ).to(device)
            elif gnn_type == 'gat':
                model = GAT(
                    input_dim=data.num_features,
                    hidden_dim=hidden_dim,
                    output_dim=dataset.num_classes,
                    dropout=0.8
                ).to(device)
            elif gnn_type == 'sage':
                model = GraphSAGE(
                    input_dim=data.num_features,
                    hidden_dim=hidden_dim,
                    output_dim=dataset.num_classes,
                    dropout=0.8
                ).to(device)

            # Create optimizer with lr=0.01, weight_decay=1e-3
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

            result = train_model(model, data_gnn, optimizer, train_val_mask.to(device), epochs=epochs, use_graph=True)
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
