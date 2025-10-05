"""
Generate plots for Ioannis's experiments across all three datasets.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11


def load_experiment_data(experiment_name, datasets):
    """Load experiment results for all datasets."""
    data = {}
    for dataset in datasets:
        file_path = Path(f'results/metrics/ioannis_experiments/{experiment_name}_{dataset}.json')
        if file_path.exists():
            with open(file_path, 'r') as f:
                data[dataset] = json.load(f)
    return data


def plot_row_norm_study(save_path='results/plots/ioannis_row_norm_comparison.png'):
    """Plot row normalization study across all datasets."""
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    data = load_experiment_data('row_norm_study', datasets)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, dataset in enumerate(datasets):
        if dataset not in data:
            continue
            
        ax = axes[idx]
        
        # Extract accuracies
        mlp_raw = data[dataset]['mlp_raw']['accuracy_mean'] * 100
        mlp_scaled = data[dataset]['mlp_scaled']['accuracy_mean'] * 100
        row_raw = data[dataset]['row_norm_mlp_raw']['accuracy_mean'] * 100
        row_scaled = data[dataset]['row_norm_mlp_scaled']['accuracy_mean'] * 100
        
        mlp_raw_std = data[dataset]['mlp_raw']['accuracy_std'] * 100
        mlp_scaled_std = data[dataset]['mlp_scaled']['accuracy_std'] * 100
        row_raw_std = data[dataset]['row_norm_mlp_raw']['accuracy_std'] * 100
        row_scaled_std = data[dataset]['row_norm_mlp_scaled']['accuracy_std'] * 100
        
        # Create grouped bar chart
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, [mlp_raw, mlp_scaled], width,
                      label='Regular MLP', color='#3498db', 
                      yerr=[mlp_raw_std, mlp_scaled_std],
                      capsize=5, edgecolor='black', linewidth=1.5)
        
        bars2 = ax.bar(x + width/2, [row_raw, row_scaled], width,
                      label='RowNorm MLP', color='#e74c3c',
                      yerr=[row_raw_std, row_scaled_std],
                      capsize=5, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title(f'{dataset}', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Raw\nEigenvectors', 'Scaled\nEigenvectors'])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 100])
    
    plt.suptitle('Experiment A: Row Normalization Study', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_gcn_eigenspace(save_path='results/plots/ioannis_gcn_eigenspace.png'):
    """Plot GCN + eigenspace results across datasets."""
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    data = load_experiment_data('gcn_eigenspace', datasets)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(datasets))
    width = 0.25
    
    original = []
    raw = []
    scaled = []
    original_std = []
    raw_std = []
    scaled_std = []
    
    for dataset in datasets:
        if dataset not in data:
            continue
        original.append(data[dataset]['gcn_original']['accuracy_mean'] * 100)
        raw.append(data[dataset]['gcn_raw_eigenspace']['accuracy_mean'] * 100)
        scaled.append(data[dataset]['gcn_scaled_eigenspace']['accuracy_mean'] * 100)
        
        original_std.append(data[dataset]['gcn_original']['accuracy_std'] * 100)
        raw_std.append(data[dataset]['gcn_raw_eigenspace']['accuracy_std'] * 100)
        scaled_std.append(data[dataset]['gcn_scaled_eigenspace']['accuracy_std'] * 100)
    
    bars1 = ax.bar(x - width, original, width, label='GCN + Original Features',
                  color='#2ecc71', yerr=original_std, capsize=5,
                  edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, raw, width, label='GCN + Raw Eigenspace',
                  color='#e74c3c', yerr=raw_std, capsize=5,
                  edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, scaled, width, label='GCN + Scaled Eigenspace',
                  color='#f39c12', yerr=scaled_std, capsize=5,
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_title('Experiment B: GCN with Eigenspace Features', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    # Add annotation about PubMed difference
    ax.annotate('PubMed: Eigenspace helps GCN!\n(Different pattern)', 
                xy=(2, 84.5), xytext=(1.3, 90),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_random_splits(save_path='results/plots/ioannis_random_splits.png'):
    """Plot random split results across datasets."""
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    data = load_experiment_data('random_split', datasets)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Absolute accuracy
    x = np.arange(len(datasets))
    width = 0.25
    
    raw_mlp = []
    eigen_mlp = []
    gcn = []
    raw_std = []
    eigen_std = []
    gcn_std = []
    
    for dataset in datasets:
        if dataset not in data:
            continue
        raw_mlp.append(data[dataset]['raw_mlp']['accuracy_mean'] * 100)
        eigen_mlp.append(data[dataset]['eigenspace_mlp']['accuracy_mean'] * 100)
        gcn.append(data[dataset]['gcn']['accuracy_mean'] * 100)
        
        raw_std.append(data[dataset]['raw_mlp']['accuracy_std'] * 100)
        eigen_std.append(data[dataset]['eigenspace_mlp']['accuracy_std'] * 100)
        gcn_std.append(data[dataset]['gcn']['accuracy_std'] * 100)
    
    bars1 = ax1.bar(x - width, raw_mlp, width, label='Raw MLP',
                   color='#95a5a6', yerr=raw_std, capsize=5,
                   edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x, eigen_mlp, width, label='Eigenspace MLP',
                   color='#3498db', yerr=eigen_std, capsize=5,
                   edgecolor='black', linewidth=1.5)
    bars3 = ax1.bar(x + width, gcn, width, label='GCN',
                   color='#2ecc71', yerr=gcn_std, capsize=5,
                   edgecolor='black', linewidth=1.5)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax1.set_title('(A) Absolute Accuracy - 60/20/20 Random Splits', 
                 fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    # Right plot: % of GCN performance
    pct_of_gcn = [(e/g)*100 for e, g in zip(eigen_mlp, gcn)]
    
    bars = ax2.bar(x, pct_of_gcn, width=0.5, color='#3498db',
                   edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, pct_of_gcn):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, label='GCN Performance')
    ax2.set_ylabel('Eigenspace as % of GCN', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax2.set_title('(B) Eigenspace MLP Relative Performance', 
                 fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([90, 105])
    
    plt.suptitle('Experiment C: 60/20/20 Random Splits', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_comprehensive_summary(save_path='results/plots/ioannis_complete_summary.png'):
    """Create comprehensive 3x3 summary figure."""
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Load all data
    row_norm_data = load_experiment_data('row_norm_study', datasets)
    gcn_data = load_experiment_data('gcn_eigenspace', datasets)
    split_data = load_experiment_data('random_split', datasets)
    
    # Row 1: Row normalization for each dataset
    for idx, dataset in enumerate(datasets):
        ax = fig.add_subplot(gs[0, idx])
        
        if dataset in row_norm_data:
            data = row_norm_data[dataset]
            
            mlp_raw = data['mlp_raw']['accuracy_mean'] * 100
            mlp_scaled = data['mlp_scaled']['accuracy_mean'] * 100
            row_raw = data['row_norm_mlp_raw']['accuracy_mean'] * 100
            row_scaled = data['row_norm_mlp_scaled']['accuracy_mean'] * 100
            
            x = np.arange(2)
            width = 0.35
            
            ax.bar(x - width/2, [mlp_raw, mlp_scaled], width,
                  label='Regular MLP', color='#3498db')
            ax.bar(x + width/2, [row_raw, row_scaled], width,
                  label='RowNorm MLP', color='#e74c3c')
            
            ax.set_title(f'{dataset}\nRow Normalization', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(['Raw', 'Scaled'], fontsize=9)
            ax.set_ylabel('Accuracy (%)', fontweight='bold')
            if idx == 0:
                ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
    
    # Row 2: GCN + eigenspace for each dataset
    for idx, dataset in enumerate(datasets):
        ax = fig.add_subplot(gs[1, idx])
        
        if dataset in gcn_data:
            data = gcn_data[dataset]
            
            original = data['gcn_original']['accuracy_mean'] * 100
            raw = data['gcn_raw_eigenspace']['accuracy_mean'] * 100
            scaled = data['gcn_scaled_eigenspace']['accuracy_mean'] * 100
            
            x = np.arange(3)
            
            colors = ['#2ecc71', '#e74c3c', '#f39c12']
            bars = ax.bar(x, [original, raw, scaled], color=colors,
                         edgecolor='black', linewidth=1.2)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
            
            ax.set_title(f'{dataset}\nGCN + Eigenspace', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(['Original', 'Raw\nEigen', 'Scaled\nEigen'], fontsize=9)
            ax.set_ylabel('Accuracy (%)', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Random splits for each dataset
    for idx, dataset in enumerate(datasets):
        ax = fig.add_subplot(gs[2, idx])
        
        if dataset in split_data:
            data = split_data[dataset]
            
            raw_mlp = data['raw_mlp']['accuracy_mean'] * 100
            eigen_mlp = data['eigenspace_mlp']['accuracy_mean'] * 100
            gcn = data['gcn']['accuracy_mean'] * 100
            
            x = np.arange(3)
            
            colors = ['#95a5a6', '#3498db', '#2ecc71']
            bars = ax.bar(x, [raw_mlp, eigen_mlp, gcn], color=colors,
                         edgecolor='black', linewidth=1.2)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
            
            pct = (eigen_mlp / gcn) * 100
            ax.set_title(f'{dataset}\n60/20/20 Split ({pct:.1f}% of GCN)', 
                        fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(['Raw\nMLP', 'Eigenspace\nMLP', 'GCN'], fontsize=9)
            ax.set_ylabel('Accuracy (%)', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Complete Summary: All Experiments Across Datasets", 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_eigenvalue_scaling_impact(save_path='results/plots/ioannis_eigenvalue_impact.png'):
    """Plot the impact of eigenvalue scaling across datasets."""
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    data = load_experiment_data('row_norm_study', datasets)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(datasets))
    width = 0.4
    
    improvements_mlp = []
    improvements_row = []
    
    for dataset in datasets:
        if dataset not in data:
            continue
        
        # Regular MLP improvement
        mlp_raw = data[dataset]['mlp_raw']['accuracy_mean']
        mlp_scaled = data[dataset]['mlp_scaled']['accuracy_mean']
        improvements_mlp.append((mlp_scaled - mlp_raw) * 100)
        
        # RowNorm MLP improvement
        row_raw = data[dataset]['row_norm_mlp_raw']['accuracy_mean']
        row_scaled = data[dataset]['row_norm_mlp_scaled']['accuracy_mean']
        improvements_row.append((row_scaled - row_raw) * 100)
    
    bars1 = ax.bar(x - width/2, improvements_mlp, width,
                  label='Regular MLP', color='#3498db',
                  edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, improvements_row, width,
                  label='RowNorm MLP', color='#e74c3c',
                  edgecolor='black', linewidth=1.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Improvement from Eigenvalue Scaling (%)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Eigenvalue Scaling (1/(λ+0.1))', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate PubMed anomaly
    ax.annotate('PubMed: Minimal benefit\nfrom eigenvalue scaling', 
                xy=(2, 0.5), xytext=(1.5, 6),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Create output directory
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING IOANNIS EXPERIMENT PLOTS")
    print("="*70 + "\n")
    
    print("Creating row normalization comparison...")
    plot_row_norm_study()
    
    print("Creating GCN + eigenspace comparison...")
    plot_gcn_eigenspace()
    
    print("Creating random splits comparison...")
    plot_random_splits()
    
    print("Creating eigenvalue scaling impact plot...")
    plot_eigenvalue_scaling_impact()
    
    print("Creating comprehensive summary...")
    plot_comprehensive_summary()
    
    print("\n" + "="*70)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  - results/plots/ioannis_row_norm_comparison.png")
    print("  - results/plots/ioannis_gcn_eigenspace.png")
    print("  - results/plots/ioannis_random_splits.png")
    print("  - results/plots/ioannis_eigenvalue_impact.png")
    print("  - results/plots/ioannis_complete_summary.png")
    print("\n")
