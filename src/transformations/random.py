import numpy as np
from .base import BaseTransformation


class RandomTransformation(BaseTransformation):
    """Random projection transformation (baseline)."""
    
    def __init__(self, target_dim=None, distribution='gaussian', seed=42):
        """
        Args:
            target_dim: Target dimension (if None, keeps original dimension)
            distribution: 'gaussian', 'sparse', 'orthogonal'
            seed: Random seed
        """
        self.target_dim = target_dim
        self.distribution = distribution
        self.seed = seed
        self.projection_matrix = None
    
    def fit_transform(self, X, L_norm=None):
        """
        Random projection transformation.
        
        Args:
            X: Feature matrix (N × D)
            L_norm: Not used (for API compatibility)
        
        Returns:
            X_transformed: Transformed features (N × K)
        """
        np.random.seed(self.seed)
        N, D = X.shape
        K = self.target_dim if self.target_dim is not None else D
        
        print(f"[Random] Computing random projection...")
        print(f"[Random] Input shape: {X.shape}, Distribution: {self.distribution}")
        
        if self.distribution == 'gaussian':
            # Gaussian random projection
            self.projection_matrix = np.random.randn(D, K) / np.sqrt(K)
        
        elif self.distribution == 'sparse':
            # Sparse random projection (Achlioptas)
            self.projection_matrix = np.random.choice(
                [-1, 0, 1], size=(D, K), p=[1/6, 2/3, 1/6]
            ) / np.sqrt(K)
        
        elif self.distribution == 'orthogonal':
            # Random orthogonal projection
            random_matrix = np.random.randn(D, D)
            Q, _ = np.linalg.qr(random_matrix)
            self.projection_matrix = Q[:, :K]
        
        X_transformed = X @ self.projection_matrix
        
        print(f"[Random] Output shape: {X_transformed.shape}")
        
        return X_transformed
