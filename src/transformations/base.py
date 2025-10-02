from abc import ABC, abstractmethod


class BaseTransformation(ABC):
    """Base class for feature transformations."""
    
    @abstractmethod
    def fit_transform(self, X, L_norm=None):
        """
        Transform features.
        
        Args:
            X: Feature matrix (N × D)
            L_norm: Normalized Laplacian (optional, N × N)
        
        Returns:
            X_transformed: Transformed features (N × K)
        """
        pass
