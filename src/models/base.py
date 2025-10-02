import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for all models."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x, edge_index=None):
        raise NotImplementedError
    
    def reset_parameters(self):
        """Reset all parameters."""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
