
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from .base import BaseModel


class GAT(BaseModel):
    """Graph Attention Network."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 num_heads=8, dropout=0.6):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, 
                                     heads=num_heads, dropout=dropout))
        
        # Output layer (single head)
        self.convs.append(GATConv(hidden_dim * num_heads, output_dim, 
                                 heads=1, concat=False, dropout=dropout))
    
    def forward(self, x, edge_index):
        """Forward pass."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
