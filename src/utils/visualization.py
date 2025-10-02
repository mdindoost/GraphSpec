
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_dimensionality_results(results, dataset_name, save_path=None):
    """
    Plot dimensionality experiment results.
    
    Args:
        results: Dictionary with results for each dimension
        dataset_name: Name of dataset
        save_path: Path to save figure
    """
    dimensions = sorted([int(d) for d in results.keys()])
    random_means = [results[str(d)]['random_mean'] for d in dimensions]
    random_stds = [results[str(d)]['random_std'] for d in dimensions]
    eigen_means = [results[str(d)]['eigenspace_mean'] for d in dimensions]
    eigen_stds = [results[str(d)]['eigenspace_std'] for d in dimensions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines with error bars
    ax.errorbar(dimensions, random_means, yerr=random_stds, 
                label='Random Projection', marker='o', capsize=5)
    ax.errorbar(dimensions, eigen_means, yerr=eigen_stds, 
                label='Eigenspace Projection', marker='s', capsize=5)
    
    ax.set_xlabel('Target Dimension K', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Dimensionality Analysis - {dataset_name}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_baseline_comparison(results, dataset_name, save_path=None):
    """
    Plot baseline comparison results.
    
    Args:
        results: Dictionary with results for each method
        dataset_name: Name of dataset
        save_path: Path to save figure
    """
    methods = list(results.keys())
    accuracies = [results[m]['accuracy_mean'] for m in methods]
    stds = [results[m]['accuracy_std'] for m in methods]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax.bar(methods, accuracies, yerr=stds, capsize=5, 
                  color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Baseline Comparison - {dataset_name}', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc, std in zip(bars, accuracies, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{acc:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_gnn_comparison(results, dataset_name, save_path=None):
    """
    Plot GNN comparison results.
    
    Args:
        results: Dictionary with results for each model
        dataset_name: Name of dataset
        save_path: Path to save figure
    """
    models = list(results.keys())
    accuracies = [results[m]['accuracy_mean'] for m in models]
    stds = [results[m]['accuracy_std'] for m in models]
    times = [results[m]['time_mean'] for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars1 = ax1.bar(models, accuracies, yerr=stds, capsize=5,
                    color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title(f'Accuracy Comparison - {dataset_name}', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    
    # Time comparison
    bars2 = ax2.bar(models, times, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Training Time (s)', fontsize=12)
    ax2.set_title(f'Speed Comparison - {dataset_name}', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_eigenvalue_spectrum(eigenvalues, save_path=None):
    """
    Plot eigenvalue spectrum.
    
    Args:
        eigenvalues: Array of eigenvalues
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(len(eigenvalues)), eigenvalues, 'b-', linewidth=2)
    ax.set_xlabel('Eigenvalue Index', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title('Eigenvalue Spectrum', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
