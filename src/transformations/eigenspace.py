# src/transformations/eigenspace.py
"""
Spectral Eigenspace Transformation for Graph Feature Learning

This module implements various eigenspace projection strategies based on the
Rayleigh-Ritz procedure. The transformation projects the graph Laplacian onto
the feature space and computes eigenvectors that capture graph structure.

Key insight: Different scaling strategies emphasize different aspects of the
graph structure (smooth vs. sharp signals).
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import qr, eigh
from sklearn.preprocessing import StandardScaler


class EigenspaceTransformation:
    """
    Spectral eigenspace transformation using Rayleigh-Ritz procedure.
    
    This transformation projects the normalized graph Laplacian onto the feature
    space and computes an eigendecomposition. Different strategies for scaling
    the resulting eigenvectors lead to different emphasis on graph frequencies.
    
    Attributes:
        target_dim: Target dimension (if None, keeps original dimension)
        strategy: Scaling strategy to use (see STRATEGIES for options)
        eigenvalues: Computed eigenvalues after fit_transform
        method_name: Human-readable name of the strategy
    """
    
    # Available strategies with descriptions
    STRATEGIES = {
        'inverse_eigenvalue': 'Weight by 1/(λ+0.1) - emphasizes smooth signals (BEST)',
        'match_input_std': 'Scale to match input std - simple magnitude preservation',
        'sqrt_n': 'Scale by √N - restores magnitude from orthonormal vectors',
        'sqrt_eigenvalue': 'Weight by √λ - moderate eigenvalue emphasis',
        'no_scaling': 'No scaling - keeps orthonormal vectors (BASELINE, POOR)',
        'standardize': 'Apply StandardScaler after projection',
        'direct_weighting': 'Apply inverse weights directly to features',
    }
    
    def __init__(self, target_dim=None, strategy='inverse_eigenvalue', eigenvalue_strategy='full'):
        """
        Initialize eigenspace transformation.
        
        Args:
            target_dim: Target dimension (if None, keeps original D)
            strategy: Scaling strategy (see STRATEGIES)
            eigenvalue_strategy: 'full', 'low', 'mid', 'high' - which eigenvalues to use
        """
        self.target_dim = target_dim
        self.strategy = strategy
        self.eigenvalue_strategy = eigenvalue_strategy
        self.eigenvalues = None
        self.method_name = self.STRATEGIES.get(strategy, 'Unknown strategy')
        
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from: {list(self.STRATEGIES.keys())}")
    
    def fit_transform(self, X, L_norm):
        """
        Transform features using eigenspace projection.
        
        IMPORTANT: X should already be normalized (StandardScaler) before calling this!
        
        Args:
            X: Feature matrix (N × D), should be pre-normalized
            L_norm: Normalized Laplacian (N × N), can be sparse
        
        Returns:
            X_transformed: Transformed features (N × K)
        
        Algorithm:
            1. QR decomposition: X = QR, get orthonormal basis Q
            2. Project Laplacian: L_proj = Q^T @ L @ Q
            3. Eigendecomposition: L_proj = V Λ V^T
            4. Transform: X_new = Q @ V (with appropriate scaling)
        """
        print(f"[Eigenspace] Strategy: {self.strategy}")
        print(f"[Eigenspace] Input shape: {X.shape}")
        print(f"[Eigenspace] Input stats: mean={X.mean():.4f}, std={X.std():.4f}")
        
        # Step 1: QR decomposition to get orthonormal basis
        Q, R = qr(X, mode='economic')
        
        # Step 2: Project Laplacian onto subspace
        if sp.issparse(L_norm):
            L_proj = Q.T @ (L_norm @ Q)
        else:
            L_proj = Q.T @ L_norm @ Q
        
        # Step 3: Eigendecomposition
        eigenvalues, eigenvectors = eigh(L_proj)
        self.eigenvalues = eigenvalues
        
        # Select eigenvalues based on strategy
        if self.eigenvalue_strategy == 'low':
            # Low eigenvalues = smooth graph signals
            n_select = len(eigenvalues) // 3
            idx = np.argsort(eigenvalues)[:n_select]
        elif self.eigenvalue_strategy == 'mid':
            n_select = len(eigenvalues) // 3
            idx = np.argsort(eigenvalues)[n_select:2*n_select]
        elif self.eigenvalue_strategy == 'high':
            # High eigenvalues = sharp/noisy signals
            n_select = len(eigenvalues) // 3
            idx = np.argsort(eigenvalues)[-n_select:]
        else:  # 'full'
            idx = np.arange(len(eigenvalues))
        
        # Step 4: Apply scaling strategy
        X_transformed = self._apply_strategy(Q, eigenvectors[:, idx], eigenvalues[idx], X)
        
        # Handle target dimension
        if self.target_dim is not None and self.target_dim != X_transformed.shape[1]:
            if self.target_dim < X_transformed.shape[1]:
                X_transformed = X_transformed[:, :self.target_dim]
            else:
                padding = np.zeros((X_transformed.shape[0], self.target_dim - X_transformed.shape[1]))
                X_transformed = np.hstack([X_transformed, padding])
        
        print(f"[Eigenspace] Output shape: {X_transformed.shape}")
        print(f"[Eigenspace] Output stats: mean={X_transformed.mean():.4f}, std={X_transformed.std():.4f}")
        print(f"[Eigenspace] Eigenvalue range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
        
        return X_transformed
    
    def _apply_strategy(self, Q, eigenvectors, eigenvalues, X_original):
        """
        Apply the selected scaling strategy.
        
        Args:
            Q: Orthonormal basis from QR (N × D)
            eigenvectors: Eigenvectors (D × D or D × K)
            eigenvalues: Eigenvalues (D,) or (K,)
            X_original: Original features for reference
        
        Returns:
            X_transformed: Scaled features
        """
        if self.strategy == 'inverse_eigenvalue':
            # Weight by 1/(λ + ε) - emphasizes low eigenvalues (smooth signals)
            # This is the WINNING strategy!
            X_transformed = Q @ (eigenvectors / (eigenvalues + 0.1))
            # Scale to match input magnitude
            X_transformed = X_transformed * (X_original.std() / X_transformed.std())
        
        elif self.strategy == 'match_input_std':
            # Simple: just match the std of input
            X_transformed = Q @ eigenvectors
            X_transformed = X_transformed * (X_original.std() / X_transformed.std())
        
        elif self.strategy == 'sqrt_n':
            # Scale by sqrt(N) to restore magnitude from orthonormal vectors
            X_transformed = (Q @ eigenvectors) * np.sqrt(Q.shape[0])
        
        elif self.strategy == 'sqrt_eigenvalue':
            # Weight by sqrt(λ) - moderate eigenvalue emphasis
            X_transformed = Q @ (eigenvectors * np.sqrt(eigenvalues))
            X_transformed = X_transformed * (X_original.std() / X_transformed.std())
        
        elif self.strategy == 'no_scaling':
            # Baseline: no scaling (usually performs poorly)
            X_transformed = Q @ eigenvectors
        
        elif self.strategy == 'standardize':
            # Full standardization after projection
            X_transformed = Q @ eigenvectors
            X_transformed = X_transformed * np.sqrt(Q.shape[0])
            scaler = StandardScaler()
            X_transformed = scaler.fit_transform(X_transformed)
        
        elif self.strategy == 'direct_weighting':
            # Alternative: apply weights directly to original features
            X_transformed = X_original @ (eigenvectors / (eigenvalues + 0.1))
            X_transformed = X_transformed * (X_original.std() / X_transformed.std())
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return X_transformed
    
    def get_eigenvalue_stats(self):
        """
        Get statistics about the eigenvalues.
        
        Returns:
            dict: Statistics including min, max, mean, std, and full spectrum
        """
        if self.eigenvalues is None:
            return None
        
        return {
            'min': float(self.eigenvalues.min()),
            'max': float(self.eigenvalues.max()),
            'mean': float(self.eigenvalues.mean()),
            'std': float(self.eigenvalues.std()),
            'spectrum': self.eigenvalues.tolist(),
            'n_eigenvalues': len(self.eigenvalues)
        }
    
    @staticmethod
    def list_strategies():
        """Print all available strategies with descriptions."""
        print("\nAvailable Eigenspace Strategies:")
        print("=" * 70)
        for name, description in EigenspaceTransformation.STRATEGIES.items():
            print(f"  {name:20s}: {description}")
        print("=" * 70)
