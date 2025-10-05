# src/data/split_utils.py
"""
Utilities for creating different train/val/test splits.

Includes:
- Random 60/20/20 splits (Ioannis's suggestion)
- Public split (existing)
- Custom ratio splits
"""

import torch
import numpy as np


def create_random_split(data, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Create random train/val/test split.
    
    Args:
        data: PyG Data object
        train_ratio: Fraction of nodes for training (default: 0.6)
        val_ratio: Fraction of nodes for validation (default: 0.2)
        seed: Random seed for reproducibility
    
    Returns:
        train_mask, val_mask, test_mask (torch.BoolTensor)
    
    Example:
        >>> train_mask, val_mask, test_mask = create_random_split(data, 0.6, 0.2, 42)
        >>> print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n = data.num_nodes
    
    # Create random permutation of indices
    indices = torch.randperm(n)
    
    # Calculate split sizes
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    # Create masks
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    # Assign indices to splits
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    return train_mask, val_mask, test_mask


def create_stratified_split(data, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Create stratified random split (balanced classes in each split).
    
    Args:
        data: PyG Data object
        train_ratio: Fraction of nodes for training (default: 0.6)
        val_ratio: Fraction of nodes for validation (default: 0.2)
        seed: Random seed for reproducibility
    
    Returns:
        train_mask, val_mask, test_mask (torch.BoolTensor)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n = data.num_nodes
    num_classes = data.y.max().item() + 1
    
    # Initialize masks
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    # For each class, split proportionally
    for c in range(num_classes):
        # Get indices for this class
        class_indices = (data.y == c).nonzero(as_tuple=True)[0]
        n_class = class_indices.size(0)
        
        # Shuffle class indices
        perm = torch.randperm(n_class)
        class_indices = class_indices[perm]
        
        # Calculate split sizes for this class
        train_size = int(n_class * train_ratio)
        val_size = int(n_class * val_ratio)
        
        # Assign to splits
        train_mask[class_indices[:train_size]] = True
        val_mask[class_indices[train_size:train_size + val_size]] = True
        test_mask[class_indices[train_size + val_size:]] = True
    
    return train_mask, val_mask, test_mask


def get_public_split(data):
    """
    Return the public split from the dataset.
    
    Args:
        data: PyG Data object
    
    Returns:
        train_mask, val_mask, test_mask (torch.BoolTensor)
    """
    return data.train_mask, data.val_mask, data.test_mask


def print_split_stats(data, train_mask, val_mask, test_mask):
    """
    Print statistics about a data split.
    
    Args:
        data: PyG Data object
        train_mask: Training mask
        val_mask: Validation mask
        test_mask: Test mask
    """
    n = data.num_nodes
    num_classes = data.y.max().item() + 1
    
    print(f"\nSplit Statistics:")
    print(f"{'='*60}")
    print(f"Total nodes: {n}")
    print(f"Train: {train_mask.sum().item()} ({train_mask.sum().item()/n*100:.1f}%)")
    print(f"Val:   {val_mask.sum().item()} ({val_mask.sum().item()/n*100:.1f}%)")
    print(f"Test:  {test_mask.sum().item()} ({test_mask.sum().item()/n*100:.1f}%)")
    
    # Per-class distribution
    print(f"\nPer-class distribution:")
    print(f"{'Class':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
    print(f"{'-'*60}")
    
    for c in range(num_classes):
        class_mask = (data.y == c)
        train_count = (class_mask & train_mask).sum().item()
        val_count = (class_mask & val_mask).sum().item()
        test_count = (class_mask & test_mask).sum().item()
        
        print(f"{c:<8} {train_count:<8} {val_count:<8} {test_count:<8}")
    
    print(f"{'='*60}\n")


def create_multiple_random_splits(data, num_splits=10, train_ratio=0.6, val_ratio=0.2, 
                                  base_seed=42, stratified=False):
    """
    Create multiple random splits for cross-validation style experiments.
    
    Args:
        data: PyG Data object
        num_splits: Number of different splits to create
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        base_seed: Base random seed (each split uses base_seed + i)
        stratified: Whether to use stratified splitting
    
    Returns:
        List of (train_mask, val_mask, test_mask) tuples
    """
    splits = []
    
    for i in range(num_splits):
        seed = base_seed + i
        
        if stratified:
            masks = create_stratified_split(data, train_ratio, val_ratio, seed)
        else:
            masks = create_random_split(data, train_ratio, val_ratio, seed)
        
        splits.append(masks)
    
    return splits


if __name__ == "__main__":
    # Test split utilities
    from torch_geometric.datasets import Planetoid
    
    print("Testing split utilities...")
    
    # Load Cora
    dataset = Planetoid(root='data/raw', name='Cora', split='public')
    data = dataset[0]
    
    print(f"\nDataset: Cora")
    print(f"Nodes: {data.num_nodes}")
    print(f"Classes: {dataset.num_classes}")
    
    # Test public split
    print("\n" + "="*60)
    print("PUBLIC SPLIT (Original)")
    print("="*60)
    train_mask, val_mask, test_mask = get_public_split(data)
    print_split_stats(data, train_mask, val_mask, test_mask)
    
    # Test 60/20/20 random split
    print("\n" + "="*60)
    print("60/20/20 RANDOM SPLIT")
    print("="*60)
    train_mask, val_mask, test_mask = create_random_split(data, 0.6, 0.2, 42)
    print_split_stats(data, train_mask, val_mask, test_mask)
    
    # Test stratified split
    print("\n" + "="*60)
    print("60/20/20 STRATIFIED SPLIT")
    print("="*60)
    train_mask, val_mask, test_mask = create_stratified_split(data, 0.6, 0.2, 42)
    print_split_stats(data, train_mask, val_mask, test_mask)
    
    # Test multiple splits
    print("\n" + "="*60)
    print("MULTIPLE SPLITS (for cross-validation)")
    print("="*60)
    splits = create_multiple_random_splits(data, num_splits=3, stratified=True)
    for i, (train_mask, val_mask, test_mask) in enumerate(splits):
        print(f"\nSplit {i+1}: Train={train_mask.sum()}, "
              f"Val={val_mask.sum()}, Test={test_mask.sum()}")
    
    print("\n✓ Split utilities working correctly!")
