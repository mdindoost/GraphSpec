import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .base import BaseModel


class GCN(BaseModel):
    """Graph Convolutional Network."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
    
    def forward(self, x, edge_index):
        """Forward pass."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

