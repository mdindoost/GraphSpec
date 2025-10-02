# src/models/mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel


class MLP(BaseModel):
    """Multi-layer Perceptron for node classification."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 dropout=0.5, activation='relu'):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Activation
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            self.activation = F.relu
    
    def forward(self, x, edge_index=None):
        """
        Forward pass.
        
        Args:
            x: Node features (N × D)
            edge_index: Not used (for API compatibility with GNNs)
        
        Returns:
            log_probs: Log probabilities (N × C)
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=1)
