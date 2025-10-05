# src/models/row_norm_mlp.py
"""
MLP with L2 Row Normalization

Ioannis's suggestion: Normalize each row (feature vector) to unit L2 norm
before the linear layer and after the ReLU activation.

This tests whether row normalization provides an alternative regularization
that helps raw eigenvectors perform better.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RowNormMLP(nn.Module):
    """
    2-layer MLP with L2 row normalization.
    
    Architecture:
        Input (D) 
        -> L2 Normalize (row-wise)
        -> Linear(D, hidden_dim) 
        -> ReLU 
        -> L2 Normalize (row-wise)
        -> Dropout
        -> Linear(hidden_dim, output_dim) 
        -> LogSoftmax
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Number of output classes
        dropout: Dropout probability (default: 0.8)
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.8):
        super(RowNormMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout
        
        # Layers
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass with row normalization.
        
        Args:
            x: Input features (N, D)
        
        Returns:
            Log probabilities (N, C)
        """
        # Step 1: Normalize rows to L2 norm = 1 BEFORE linear layer
        # Each row vector will have ||x_i|| = 1
        x = F.normalize(x, p=2, dim=1)
        
        # Step 2: Linear transformation
        x = self.linear1(x)
        
        # Step 3: ReLU activation
        x = F.relu(x)
        
        # Step 4: Normalize rows AFTER ReLU
        # This ensures hidden representations also have unit norm
        x = F.normalize(x, p=2, dim=1)
        
        # Step 5: Dropout for regularization
        x = self.dropout(x)
        
        # Step 6: Output layer
        x = self.linear2(x)
        
        # Step 7: Log-softmax for classification
        return F.log_softmax(x, dim=1)
    
    def __repr__(self):
        return (f'RowNormMLP(input_dim={self.input_dim}, '
                f'hidden_dim={self.hidden_dim}, '
                f'output_dim={self.output_dim}, '
                f'dropout={self.dropout_p})')


class RowNormMLPNoDropout(nn.Module):
    """
    Row normalization MLP without dropout (for comparison).
    
    Same as RowNormMLP but without dropout layer.
    Useful for testing if row normalization alone provides regularization.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RowNormMLPNoDropout, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Normalize before linear
        x = F.normalize(x, p=2, dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        
        # Normalize after ReLU
        x = F.normalize(x, p=2, dim=1)
        x = self.linear2(x)
        
        return F.log_softmax(x, dim=1)
    
    def __repr__(self):
        return (f'RowNormMLPNoDropout(input_dim={self.input_dim}, '
                f'hidden_dim={self.hidden_dim}, '
                f'output_dim={self.output_dim})')


if __name__ == "__main__":
    # Test the model
    print("Testing RowNormMLP...")
    
    # Create dummy data
    batch_size = 10
    input_dim = 100
    hidden_dim = 64
    output_dim = 7
    
    x = torch.randn(batch_size, input_dim)
    
    # Test regular version
    model = RowNormMLP(input_dim, hidden_dim, output_dim, dropout=0.8)
    print(f"\n{model}")
    
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output is log probabilities: {output.exp().sum(dim=1)}")
    
    # Test no-dropout version
    model_no_dropout = RowNormMLPNoDropout(input_dim, hidden_dim, output_dim)
    print(f"\n{model_no_dropout}")
    
    output2 = model_no_dropout(x)
    print(f"Output shape (no dropout): {output2.shape}")
    
    print("\n✓ RowNormMLP implementation complete!")
