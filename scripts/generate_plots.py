"""
Generate all plots for GraphSpec results.

This script creates comprehensive visualizations from dimensionality and baseline results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_dimensionality_data(dataset_name):
    """Load dimensionality results from JSON."""
    file_path = Path(f'results/metrics/dimensionality_{dataset_name}.json')
    with open(file_path, 'r') as f:
        raw_data = json.load(f)
    
    # Parse the structure: keys are dimension strings like "358", "716", etc.
    dimensions = sorted([int(k) for k in raw_data.keys()])
    
    eigenspace_mean = [raw_data[str(d)]['eigenspace_mean'] for d in dimensions]
    eigenspace_std = [raw_data[str(d)]['eigenspace_std'] for d in dimensions]
    random_mean = [raw_data[str(d)]['random_mean'] for d in dimensions]
    random_std = [raw_data[str(d)]['random_std'] for d in dimensions]
    
    return {
        'dimensions': dimensions,
        'results': {
            'eigenspace': {
                'accuracies_mean': eigenspace_mean,
                'accuracies_std': eigenspace_std
            },
            'random': {
                'accuracies_mean': random_mean,
                'accuracies_std': random_std
            }
        }
    }


def plot_dimensionality_curves(save_path='results/plots/dimensionality_curves.png'):
    """
    Plot eigenspace vs random accuracy across dimensions for all datasets.
    This is the main figure showing the inverse relationship.
    """
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    colors = {'Cora': '#e74c3c', 'CiteSeer': '#3498db', 'PubMed': '#2ecc71'}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Eigenspace accuracy
    for dataset in datasets:
        data = load_dimensionality_data(dataset)
        
        dimensions = data['dimensions']
        eigen_mean = data['results']['eigenspace']['accuracies_mean']
        eigen_std = data['results']['eigenspace']['accuracies_std']
        
        # Normalize dimensions to ratio (0.25, 0.5, 1.0, 2.0, 4.0)
        D = dimensions[2]  # The middle one is D (ratio=1.0)
        ratios = [d/D for d in dimensions]
        
        ax1.plot(ratios, eigen_mean, 'o-', linewidth=2.5, 
                label=dataset, color=colors[dataset], markersize=8)
        ax1.fill_between(ratios, 
                         np.array(eigen_mean) - np.array(eigen_std),
                         np.array(eigen_mean) + np.array(eigen_std),
                         alpha=0.2, color=colors[dataset])
    
    ax1.set_xlabel('Dimension Ratio (K/D)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Eigenspace Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title('Eigenspace: Best at Low Dimensions (D/4)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax1.set_xticklabels(['D/4', 'D/2', 'D', '2D', '4D'])
    
    # Plot 2: Random projection accuracy
    for dataset in datasets:
        data = load_dimensionality_data(dataset)
        
        dimensions = data['dimensions']
        random_mean = data['results']['random']['accuracies_mean']
        random_std = data['results']['random']['accuracies_std']
        
        D = dimensions[2]
        ratios = [d/D for d in dimensions]
        
        ax2.plot(ratios, random_mean, 's-', linewidth=2.5,
                label=dataset, color=colors[dataset], markersize=8)
        ax2.fill_between(ratios,
                         np.array(random_mean) - np.array(random_std),
                         np.array(random_mean) + np.array(random_std),
                         alpha=0.2, color=colors[dataset])
    
    ax2.set_xlabel('Dimension Ratio (K/D)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Random Projection Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title('Random: Improves with High Dimensions', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax2.set_xticklabels(['D/4', 'D/2', 'D', '2D', '4D'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_improvement_across_dimensions(save_path='results/plots/improvement_vs_dimension.png'):
    """Plot improvement (Eigenspace - Random) across dimensions."""
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    colors = {'Cora': '#e74c3c', 'CiteSeer': '#3498db', 'PubMed': '#2ecc71'}
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for dataset in datasets:
        data = load_dimensionality_data(dataset)
        
        dimensions = data['dimensions']
        eigen_mean = np.array(data['results']['eigenspace']['accuracies_mean'])
        random_mean = np.array(data['results']['random']['accuracies_mean'])
        
        improvement = (eigen_mean - random_mean) * 100  # Percentage points
        
        D = dimensions[2]
        ratios = [d/D for d in dimensions]
        
        ax.plot(ratios, improvement, 'o-', linewidth=2.5, 
               label=dataset, color=colors[dataset], markersize=10)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Dimension Ratio (K/D)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Improvement (Eigenspace - Random) %', fontsize=13, fontweight='bold')
    ax.set_title('Eigenspace Advantage Decreases with Dimension', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax.set_xticklabels(['D/4', 'D/2', 'D', '2D', '4D'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_optimal_dimension_comparison(save_path='results/plots/optimal_dimension_bar.png'):
    """Bar chart comparing D/4 vs D for all datasets."""
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    
    # Extract D/4 and D results from dimensionality data
    results = {}
    for dataset in datasets:
        data = load_dimensionality_data(dataset)
        dimensions = data['dimensions']
        D_index = 2  # Middle value is D (1.0)
        D4_index = 0  # First value is D/4 (0.25)
        
        results[dataset] = {
            'eigenspace_D4': data['results']['eigenspace']['accuracies_mean'][D4_index],
            'eigenspace_D': data['results']['eigenspace']['accuracies_mean'][D_index],
            'random_D4': data['results']['random']['accuracies_mean'][D4_index],
            'random_D': data['results']['random']['accuracies_mean'][D_index],
        }
    
    # Create grouped bar chart
    x = np.arange(len(datasets))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    eigen_d4 = [results[d]['eigenspace_D4'] * 100 for d in datasets]
    eigen_d = [results[d]['eigenspace_D'] * 100 for d in datasets]
    random_d4 = [results[d]['random_D4'] * 100 for d in datasets]
    random_d = [results[d]['random_D'] * 100 for d in datasets]
    
    bars1 = ax.bar(x - 1.5*width, eigen_d4, width, label='Eigenspace @ D/4', 
                   color='#27ae60', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x - 0.5*width, eigen_d, width, label='Eigenspace @ D',
                   color='#95a5a6', edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + 0.5*width, random_d4, width, label='Random @ D/4',
                   color='#e67e22', edgecolor='black', linewidth=1.5)
    bars4 = ax.bar(x + 1.5*width, random_d, width, label='Random @ D',
                   color='#c0392b', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('D/4 vs D Comparison: Eigenspace Benefits from Compression', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_combined_summary(save_path='results/plots/complete_summary.png'):
    """Create a comprehensive 4-panel summary figure."""
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    colors = {'Cora': '#e74c3c', 'CiteSeer': '#3498db', 'PubMed': '#2ecc71'}
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Eigenspace curves
    ax1 = fig.add_subplot(gs[0, 0])
    for dataset in datasets:
        data = load_dimensionality_data(dataset)
        dimensions = data['dimensions']
        eigen_mean = data['results']['eigenspace']['accuracies_mean']
        D = dimensions[2]
        ratios = [d/D for d in dimensions]
        
        ax1.plot(ratios, eigen_mean, 'o-', linewidth=2.5,
                label=dataset, color=colors[dataset], markersize=8)
    
    ax1.set_xlabel('Dimension Ratio (K/D)', fontweight='bold')
    ax1.set_ylabel('Eigenspace Accuracy', fontweight='bold')
    ax1.set_title('(A) Eigenspace: Optimal at D/4', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax1.set_xticklabels(['D/4', 'D/2', 'D', '2D', '4D'])
    
    # Panel 2: Random curves
    ax2 = fig.add_subplot(gs[0, 1])
    for dataset in datasets:
        data = load_dimensionality_data(dataset)
        dimensions = data['dimensions']
        random_mean = data['results']['random']['accuracies_mean']
        D = dimensions[2]
        ratios = [d/D for d in dimensions]
        
        ax2.plot(ratios, random_mean, 's-', linewidth=2.5,
                label=dataset, color=colors[dataset], markersize=8)
    
    ax2.set_xlabel('Dimension Ratio (K/D)', fontweight='bold')
    ax2.set_ylabel('Random Projection Accuracy', fontweight='bold')
    ax2.set_title('(B) Random: Needs High Dimensions', fontweight='bold', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax2.set_xticklabels(['D/4', 'D/2', 'D', '2D', '4D'])
    
    # Panel 3: Improvement across dimensions
    ax3 = fig.add_subplot(gs[1, 0])
    for dataset in datasets:
        data = load_dimensionality_data(dataset)
        dimensions = data['dimensions']
        eigen_mean = np.array(data['results']['eigenspace']['accuracies_mean'])
        random_mean = np.array(data['results']['random']['accuracies_mean'])
        improvement = (eigen_mean - random_mean) * 100
        D = dimensions[2]
        ratios = [d/D for d in dimensions]
        
        ax3.plot(ratios, improvement, 'o-', linewidth=2.5,
                label=dataset, color=colors[dataset], markersize=8)
    
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax3.set_xlabel('Dimension Ratio (K/D)', fontweight='bold')
    ax3.set_ylabel('Improvement (%)', fontweight='bold')
    ax3.set_title('(C) Eigenspace Advantage Peaks at D/4', fontweight='bold', fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax3.set_xticklabels(['D/4', 'D/2', 'D', '2D', '4D'])
    
    # Panel 4: D/4 vs D bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    results = {}
    for dataset in datasets:
        data = load_dimensionality_data(dataset)
        D_index = 2
        D4_index = 0
        results[dataset] = {
            'eigenspace_D4': data['results']['eigenspace']['accuracies_mean'][D4_index],
            'eigenspace_D': data['results']['eigenspace']['accuracies_mean'][D_index],
        }
    
    x = np.arange(len(datasets))
    width = 0.35
    
    eigen_d4 = [results[d]['eigenspace_D4'] * 100 for d in datasets]
    eigen_d = [results[d]['eigenspace_D'] * 100 for d in datasets]
    
    bars1 = ax4.bar(x - width/2, eigen_d4, width, label='D/4 (Compressed)',
                    color='#27ae60', edgecolor='black', linewidth=1.5)
    bars2 = ax4.bar(x + width/2, eigen_d, width, label='D (Full)',
                    color='#95a5a6', edgecolor='black', linewidth=1.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax4.set_xlabel('Dataset', fontweight='bold')
    ax4.set_ylabel('Eigenspace Accuracy (%)', fontweight='bold')
    ax4.set_title('(D) Compression Improves Accuracy', fontweight='bold', fontsize=13)
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('GraphSpec: Complete Results Summary', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_results_table(save_path='results/plots/results_table.png'):
    """Create a visual table of key results."""
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    
    # Collect data
    table_data = []
    for dataset in datasets:
        data = load_dimensionality_data(dataset)
        D_index = 2
        D4_index = 0
        
        eigen_d4 = data['results']['eigenspace']['accuracies_mean'][D4_index] * 100
        eigen_d = data['results']['eigenspace']['accuracies_mean'][D_index] * 100
        random_d4 = data['results']['random']['accuracies_mean'][D4_index] * 100
        
        improvement = eigen_d4 - random_d4
        compression_gain = eigen_d4 - eigen_d
        
        table_data.append([
            dataset,
            f"{eigen_d4:.2f}%",
            f"{eigen_d:.2f}%",
            f"+{compression_gain:.2f}%",
            f"+{improvement:.2f}%"
        ])
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    headers = ['Dataset', 'Eigenspace\n@ D/4', 'Eigenspace\n@ D', 
               'Gain from\nCompression', 'Improvement\nover Random']
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.2, 0.2, 0.2, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if j == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    plt.title('Key Results Summary: D/4 Compression Benefits', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Create output directory
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING GRAPHSPEC PLOTS")
    print("="*70 + "\n")
    
    # Generate all plots
    print("📊 Creating dimensionality curves...")
    plot_dimensionality_curves()
    
    print("📈 Creating improvement plot...")
    plot_improvement_across_dimensions()
    
    print("📊 Creating optimal dimension comparison...")
    plot_optimal_dimension_comparison()
    
    print("🎨 Creating combined summary figure...")
    plot_combined_summary()
    
    print("📋 Creating results table...")
    create_results_table()
    
    print("\n" + "="*70)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  - results/plots/dimensionality_curves.png")
    print("  - results/plots/improvement_vs_dimension.png")
    print("  - results/plots/optimal_dimension_bar.png")
    print("  - results/plots/complete_summary.png")
    print("  - results/plots/results_table.png")
    print("\n")
