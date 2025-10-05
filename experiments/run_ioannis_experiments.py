"""
Ioannis's Suggested Experiments

Three main investigations:
1. Row Normalization Study - Does row normalization help raw eigenvectors?
2. GCN with Eigenspace Features - Does GCN benefit from eigenspace preprocessing?
3. 60/20/20 Random Splits - Do results generalize beyond public split?
"""

import sys
import os
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.append('.')

import torch
import torch.nn.functional as F
import numpy as np
import time
import argparse
import json
from torch_geometric.datasets import Planetoid
from sklearn.preprocessing import StandardScaler

from src.transformations.eigenspace import EigenspaceTransformation
from src.data.graph_utils import compute_normalized_laplacian
from src.models.mlp import MLP
from src.models.row_norm_mlp import RowNormMLP
from src.models.gcn import GCN
from src.data.split_utils import create_random_split, print_split_stats


def train_model(model, data, optimizer, train_mask, epochs=500, use_graph=False):
    """Train model and return metrics (same as run_baseline.py)."""
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
                    best_test_f1 = test_acc
                    patience = 0
                else:
                    patience += 1

                if patience >= 30:
                    break

    train_time = time.time() - start_time

    return {
        'test_acc': best_test_acc,
        'test_f1_micro': best_test_f1,
        'test_f1_macro': best_test_f1,
        'train_time': train_time
    }


def experiment_a_row_normalization(dataset_name='Cora', hidden_dim=64, epochs=500, 
                                   num_runs=10, device='cpu'):
    """
    Experiment A: Row Normalization Study
    
    Compare 4 configurations:
    1. Regular MLP + raw eigenvectors (no eigenvalue scaling)
    2. Regular MLP + scaled eigenvectors (1/(λ+0.1))
    3. RowNormMLP + raw eigenvectors
    4. RowNormMLP + scaled eigenvectors
    
    Question: Does row normalization help raw eigenvectors perform better?
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT A: ROW NORMALIZATION STUDY - {dataset_name}")
    print("="*80)
    print("\nTesting if row normalization helps raw eigenvectors...")
    
    # Load dataset
    dataset = Planetoid(root='data/raw', name=dataset_name, split='public')
    data = dataset[0]
    train_val_mask = data.train_mask | data.val_mask
    
    print(f"\nDataset: {dataset_name}")
    print(f"  Train+Val: {train_val_mask.sum()} samples")
    print(f"  Test: {data.test_mask.sum()} samples")
    print(f"  Using D/4 compression (optimal)\n")
    
    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(data.x.numpy())
    
    # Compute Laplacian
    edge_index_np = data.edge_index.numpy()
    L_norm = compute_normalized_laplacian(edge_index_np, data.num_nodes)
    
    # Compute eigenspace transformations
    target_dim = data.num_features // 4  # D/4
    
    print(f"Computing eigenspace features...")
    
    # Raw eigenvectors (no eigenvalue scaling)
    transform_raw = EigenspaceTransformation(target_dim=target_dim, strategy='no_scaling')
    X_raw = transform_raw.fit_transform(X_normalized, L_norm)
    
    # Scaled eigenvectors (with eigenvalue weighting)
    transform_scaled = EigenspaceTransformation(target_dim=target_dim, strategy='inverse_eigenvalue')
    X_scaled = transform_scaled.fit_transform(X_normalized, L_norm)
    
    print(f"  Raw eigenspace shape: {X_raw.shape}")
    print(f"  Scaled eigenspace shape: {X_scaled.shape}\n")
    
    # Storage for results
    all_results = {
        'mlp_raw': [],
        'mlp_scaled': [],
        'row_norm_mlp_raw': [],
        'row_norm_mlp_scaled': []
    }
    
    # Run multiple times
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        print("-" * 60)
        
        # 1. Regular MLP + raw eigenvectors
        print("[1/4] Regular MLP + raw eigenvectors...")
        data_raw = data.clone()
        data_raw.x = torch.FloatTensor(X_raw)
        
        model = MLP(X_raw.shape[1], hidden_dim, dataset.num_classes, dropout=0.8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
        result = train_model(model, data_raw, optimizer, train_val_mask, epochs=epochs)
        all_results['mlp_raw'].append(result)
        print(f"   Test Acc: {result['test_acc']:.4f}")
        
        # 2. Regular MLP + scaled eigenvectors
        print("[2/4] Regular MLP + scaled eigenvectors...")
        data_scaled = data.clone()
        data_scaled.x = torch.FloatTensor(X_scaled)
        
        model = MLP(X_scaled.shape[1], hidden_dim, dataset.num_classes, dropout=0.8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
        result = train_model(model, data_scaled, optimizer, train_val_mask, epochs=epochs)
        all_results['mlp_scaled'].append(result)
        print(f"   Test Acc: {result['test_acc']:.4f}")
        
        # 3. RowNormMLP + raw eigenvectors
        print("[3/4] RowNormMLP + raw eigenvectors...")
        data_raw = data.clone()
        data_raw.x = torch.FloatTensor(X_raw)
        
        model = RowNormMLP(X_raw.shape[1], hidden_dim, dataset.num_classes, dropout=0.8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
        result = train_model(model, data_raw, optimizer, train_val_mask, epochs=epochs)
        all_results['row_norm_mlp_raw'].append(result)
        print(f"   Test Acc: {result['test_acc']:.4f}")
        
        # 4. RowNormMLP + scaled eigenvectors
        print("[4/4] RowNormMLP + scaled eigenvectors...")
        data_scaled = data.clone()
        data_scaled.x = torch.FloatTensor(X_scaled)
        
        model = RowNormMLP(X_scaled.shape[1], hidden_dim, dataset.num_classes, dropout=0.8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
        result = train_model(model, data_scaled, optimizer, train_val_mask, epochs=epochs)
        all_results['row_norm_mlp_scaled'].append(result)
        print(f"   Test Acc: {result['test_acc']:.4f}\n")
    
    # Aggregate results
    print("\n" + "="*80)
    print(f"RESULTS SUMMARY - ROW NORMALIZATION STUDY ({num_runs} runs)")
    print("="*80)
    
    print(f"\n{'Configuration':<35} {'Accuracy':>15} {'Time (s)':>12}")
    print("-"*80)
    
    summary = {}
    for method_name, results in all_results.items():
        accs = [r['test_acc'] for r in results]
        times = [r['train_time'] for r in results]
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_time = np.mean(times)
        
        display_name = method_name.replace('_', ' ').title()
        print(f"{display_name:<35} {mean_acc:>7.4f}±{std_acc:.4f} {mean_time:>12.2f}")
        
        summary[method_name] = {
            'accuracy_mean': float(mean_acc),
            'accuracy_std': float(std_acc),
            'time_mean': float(mean_time),
            'all_results': results
        }
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    mlp_raw = summary['mlp_raw']['accuracy_mean']
    mlp_scaled = summary['mlp_scaled']['accuracy_mean']
    row_raw = summary['row_norm_mlp_raw']['accuracy_mean']
    row_scaled = summary['row_norm_mlp_scaled']['accuracy_mean']
    
    print(f"1. Regular MLP: Scaled ({mlp_scaled:.4f}) vs Raw ({mlp_raw:.4f}) = "
          f"{(mlp_scaled-mlp_raw)*100:+.2f}%")
    print(f"2. RowNorm MLP: Scaled ({row_scaled:.4f}) vs Raw ({row_raw:.4f}) = "
          f"{(row_scaled-row_raw)*100:+.2f}%")
    print(f"3. Raw eigenvectors: RowNorm ({row_raw:.4f}) vs Regular ({mlp_raw:.4f}) = "
          f"{(row_raw-mlp_raw)*100:+.2f}%")
    print(f"4. Scaled eigenvectors: RowNorm ({row_scaled:.4f}) vs Regular ({mlp_scaled:.4f}) = "
          f"{(row_scaled-mlp_scaled)*100:+.2f}%")
    
    # Save results
    output_dir = Path('results/metrics/ioannis_experiments')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'row_norm_study_{dataset_name}.json'
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_file}\n")
    
    return summary


def experiment_b_gcn_eigenspace(dataset_name='Cora', hidden_dim=64, epochs=500,
                                num_runs=10, device='cpu'):
    """
    Experiment B: GCN with Eigenspace Features
    
    Compare 3 configurations:
    1. GCN + original features (baseline)
    2. GCN + raw eigenspace features (no eigenvalue scaling)
    3. GCN + scaled eigenspace features (with eigenvalue scaling)
    
    Question: Does GCN benefit from eigenspace preprocessing, or is it redundant?
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT B: GCN WITH EIGENSPACE FEATURES - {dataset_name}")
    print("="*80)
    print("\nTesting if GCN benefits from eigenspace preprocessing...")
    
    # Load dataset
    dataset = Planetoid(root='data/raw', name=dataset_name, split='public')
    data = dataset[0]
    train_val_mask = data.train_mask | data.val_mask
    
    print(f"\nDataset: {dataset_name}")
    print(f"  Train+Val: {train_val_mask.sum()} samples")
    print(f"  Test: {data.test_mask.sum()} samples")
    print(f"  Using D/4 compression\n")
    
    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(data.x.numpy())
    
    # Compute Laplacian
    edge_index_np = data.edge_index.numpy()
    L_norm = compute_normalized_laplacian(edge_index_np, data.num_nodes)
    
    # Compute eigenspace transformations
    target_dim = data.num_features // 4
    
    print(f"Computing eigenspace features...")
    
    # Raw eigenspace
    transform_raw = EigenspaceTransformation(target_dim=target_dim, strategy='no_scaling')
    X_raw = transform_raw.fit_transform(X_normalized, L_norm)
    
    # Scaled eigenspace
    transform_scaled = EigenspaceTransformation(target_dim=target_dim, strategy='inverse_eigenvalue')
    X_scaled = transform_scaled.fit_transform(X_normalized, L_norm)
    
    print(f"  Original features: {X_normalized.shape}")
    print(f"  Eigenspace features: {X_scaled.shape}\n")
    
    # Storage for results
    all_results = {
        'gcn_original': [],
        'gcn_raw_eigenspace': [],
        'gcn_scaled_eigenspace': []
    }
    
    # Run multiple times
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        print("-" * 60)
        
        # 1. GCN + original features (baseline)
        print("[1/3] GCN + original features...")
        data_orig = data.clone()
        data_orig.x = torch.FloatTensor(X_normalized)
        
        model = GCN(data.num_features, hidden_dim, dataset.num_classes, dropout=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        result = train_model(model, data_orig, optimizer, train_val_mask, epochs=epochs, use_graph=True)
        all_results['gcn_original'].append(result)
        print(f"   Test Acc: {result['test_acc']:.4f}")
        
        # 2. GCN + raw eigenspace
        print("[2/3] GCN + raw eigenspace...")
        data_raw = data.clone()
        data_raw.x = torch.FloatTensor(X_raw)
        
        model = GCN(X_raw.shape[1], hidden_dim, dataset.num_classes, dropout=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        result = train_model(model, data_raw, optimizer, train_val_mask, epochs=epochs, use_graph=True)
        all_results['gcn_raw_eigenspace'].append(result)
        print(f"   Test Acc: {result['test_acc']:.4f}")
        
        # 3. GCN + scaled eigenspace
        print("[3/3] GCN + scaled eigenspace...")
        data_scaled = data.clone()
        data_scaled.x = torch.FloatTensor(X_scaled)
        
        model = GCN(X_scaled.shape[1], hidden_dim, dataset.num_classes, dropout=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        result = train_model(model, data_scaled, optimizer, train_val_mask, epochs=epochs, use_graph=True)
        all_results['gcn_scaled_eigenspace'].append(result)
        print(f"   Test Acc: {result['test_acc']:.4f}\n")
    
    # Aggregate results
    print("\n" + "="*80)
    print(f"RESULTS SUMMARY - GCN WITH EIGENSPACE ({num_runs} runs)")
    print("="*80)
    
    print(f"\n{'Configuration':<35} {'Accuracy':>15} {'Time (s)':>12}")
    print("-"*80)
    
    summary = {}
    for method_name, results in all_results.items():
        accs = [r['test_acc'] for r in results]
        times = [r['train_time'] for r in results]
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_time = np.mean(times)
        
        display_name = method_name.replace('_', ' ').title()
        print(f"{display_name:<35} {mean_acc:>7.4f}±{std_acc:.4f} {mean_time:>12.2f}")
        
        summary[method_name] = {
            'accuracy_mean': float(mean_acc),
            'accuracy_std': float(std_acc),
            'time_mean': float(mean_time),
            'all_results': results
        }
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    gcn_orig = summary['gcn_original']['accuracy_mean']
    gcn_raw = summary['gcn_raw_eigenspace']['accuracy_mean']
    gcn_scaled = summary['gcn_scaled_eigenspace']['accuracy_mean']
    
    print(f"1. GCN Original: {gcn_orig:.4f} (baseline)")
    print(f"2. GCN + Raw Eigenspace: {gcn_raw:.4f} ({(gcn_raw-gcn_orig)*100:+.2f}%)")
    print(f"3. GCN + Scaled Eigenspace: {gcn_scaled:.4f} ({(gcn_scaled-gcn_orig)*100:+.2f}%)")
    
    if gcn_scaled < gcn_orig:
        print(f"\n→ Eigenspace preprocessing hurts GCN (redundant spectral filtering)")
    elif gcn_scaled > gcn_orig:
        print(f"\n→ Eigenspace preprocessing helps GCN (complementary)")
    else:
        print(f"\n→ Eigenspace preprocessing has minimal effect on GCN")
    
    # Save results
    output_dir = Path('results/metrics/ioannis_experiments')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'gcn_eigenspace_{dataset_name}.json'
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_file}\n")
    
    return summary


def experiment_c_random_splits(dataset_name='Cora', hidden_dim=64, epochs=500,
                               num_runs=10, device='cpu'):
    """
    Experiment C: 60/20/20 Random Splits
    
    Re-run baseline comparison with random 60/20/20 splits instead of public split.
    
    Question: Do results generalize beyond the challenging public split?
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT C: 60/20/20 RANDOM SPLITS - {dataset_name}")
    print("="*80)
    print("\nTesting with random splits (60% train, 20% val, 20% test)...")
    
    # Load dataset
    dataset = Planetoid(root='data/raw', name=dataset_name, split='public')
    data = dataset[0]
    
    print(f"\nDataset: {dataset_name}")
    print(f"  Total nodes: {data.num_nodes}")
    print(f"  Using D/4 compression\n")
    
    # Create random split
    train_mask, val_mask, test_mask = create_random_split(data, 0.6, 0.2, seed=42)
    train_val_mask = train_mask | val_mask
    
    # Update data with new masks
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    print_split_stats(data, train_mask, val_mask, test_mask)
    
    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(data.x.numpy())
    
    # Compute Laplacian
    edge_index_np = data.edge_index.numpy()
    L_norm = compute_normalized_laplacian(edge_index_np, data.num_nodes)
    
    # Eigenspace transformation
    target_dim = data.num_features // 4
    
    print(f"Computing eigenspace features...")
    transform = EigenspaceTransformation(target_dim=target_dim, strategy='inverse_eigenvalue')
    X_eigenspace = transform.fit_transform(X_normalized, L_norm)
    print(f"  Eigenspace features: {X_eigenspace.shape}\n")
    
    # Storage for results
    all_results = {
        'raw_mlp': [],
        'eigenspace_mlp': [],
        'gcn': []
    }
    
    # Run multiple times
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        print("-" * 60)
        
        # 1. Raw MLP
        print("[1/3] Raw MLP...")
        data_raw = data.clone()
        data_raw.x = torch.FloatTensor(X_normalized)
        
        model = MLP(data.num_features, hidden_dim, dataset.num_classes, dropout=0.8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
        result = train_model(model, data_raw, optimizer, train_val_mask, epochs=epochs)
        all_results['raw_mlp'].append(result)
        print(f"   Test Acc: {result['test_acc']:.4f}")
        
        # 2. Eigenspace MLP
        print("[2/3] Eigenspace MLP...")
        data_eigen = data.clone()
        data_eigen.x = torch.FloatTensor(X_eigenspace)
        
        model = MLP(X_eigenspace.shape[1], hidden_dim, dataset.num_classes, dropout=0.8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
        result = train_model(model, data_eigen, optimizer, train_val_mask, epochs=epochs)
        all_results['eigenspace_mlp'].append(result)
        print(f"   Test Acc: {result['test_acc']:.4f}")
        
        # 3. GCN
        print("[3/3] GCN...")
        data_gcn = data.clone()
        data_gcn.x = torch.FloatTensor(X_normalized)
        
        model = GCN(data.num_features, hidden_dim, dataset.num_classes, dropout=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        result = train_model(model, data_gcn, optimizer, train_val_mask, epochs=epochs, use_graph=True)
        all_results['gcn'].append(result)
        print(f"   Test Acc: {result['test_acc']:.4f}\n")
    
    # Aggregate results
    print("\n" + "="*80)
    print(f"RESULTS SUMMARY - RANDOM 60/20/20 SPLITS ({num_runs} runs)")
    print("="*80)
    
    print(f"\n{'Method':<25} {'Accuracy':>15} {'Time (s)':>12}")
    print("-"*80)
    
    summary = {}
    for method_name, results in all_results.items():
        accs = [r['test_acc'] for r in results]
        times = [r['train_time'] for r in results]
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_time = np.mean(times)
        
        display_name = method_name.replace('_', ' ').title()
        print(f"{display_name:<25} {mean_acc:>7.4f}±{std_acc:.4f} {mean_time:>12.2f}")
        
        summary[method_name] = {
            'accuracy_mean': float(mean_acc),
            'accuracy_std': float(std_acc),
            'time_mean': float(mean_time),
            'all_results': results
        }
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    raw_acc = summary['raw_mlp']['accuracy_mean']
    eigen_acc = summary['eigenspace_mlp']['accuracy_mean']
    gcn_acc = summary['gcn']['accuracy_mean']
    
    improvement = (eigen_acc - raw_acc) * 100
    pct_of_gcn = (eigen_acc / gcn_acc) * 100
    
    print(f"1. Eigenspace vs Raw MLP: {improvement:+.2f}%")
    print(f"2. Eigenspace reaches: {pct_of_gcn:.1f}% of GCN performance")
    print(f"3. Training samples: {train_val_mask.sum()} (vs 640 in public split)")
    
    # Save results
    output_dir = Path('results/metrics/ioannis_experiments')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'random_split_{dataset_name}.json'
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_file}\n")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Ioannis's suggested experiments"
    )
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['row_norm', 'gcn_eigenspace', 'random_split', 'all'],
                       help='Which experiment to run')
    parser.add_argument('--dataset', type=str, default='Cora',
                       choices=['Cora', 'CiteSeer', 'PubMed'],
                       help='Dataset to use')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--runs', type=int, default=10,
                       help='Number of runs for averaging')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("IOANNIS'S SUGGESTED EXPERIMENTS")
    print("="*80)
    print(f"\nDataset: {args.dataset}")
    print(f"Runs: {args.runs}")
    print(f"Epochs: {args.epochs}")
    print(f"Hidden dim: {args.hidden_dim}")
    
    # Run requested experiment(s)
    if args.experiment == 'row_norm' or args.experiment == 'all':
        experiment_a_row_normalization(
            dataset_name=args.dataset,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            num_runs=args.runs,
            device=args.device
        )
    
    if args.experiment == 'gcn_eigenspace' or args.experiment == 'all':
        experiment_b_gcn_eigenspace(
            dataset_name=args.dataset,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            num_runs=args.runs,
            device=args.device
        )
    
    if args.experiment == 'random_split' or args.experiment == 'all':
        experiment_c_random_splits(
            dataset_name=args.dataset,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            num_runs=args.runs,
            device=args.device
        )
    
    print("\n" + "="*80)
    print("ALL REQUESTED EXPERIMENTS COMPLETE")
    print("="*80)
    print("\nResults saved to: results/metrics/ioannis_experiments/")
    print("\n")
