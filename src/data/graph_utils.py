
import numpy as np
import scipy.sparse as sp


def compute_normalized_laplacian(edge_index, num_nodes):
    """
    Compute normalized Laplacian: L_norm = I - D^(-1/2) A D^(-1/2)
    
    Args:
        edge_index: Edge indices (2 × E)
        num_nodes: Number of nodes
    
    Returns:
        L_norm: Normalized Laplacian (sparse matrix)
    """
    # Build adjacency matrix
    row, col = edge_index
    adj = sp.coo_matrix(
        (np.ones(len(row)), (row, col)), 
        shape=(num_nodes, num_nodes)
    )
    
    # Make symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    # Compute degree matrix
    degree = np.array(adj.sum(axis=1)).flatten()
    degree[degree == 0] = 1  # Avoid division by zero
    
    # D^(-1/2)
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(degree))
    
    # Normalized adjacency: D^(-1/2) A D^(-1/2)
    adj_normalized = d_inv_sqrt @ adj @ d_inv_sqrt
    
    # Laplacian: I - A_normalized
    identity = sp.eye(num_nodes)
    L_norm = identity - adj_normalized
    
    return L_norm.tocsr()


def compute_homophily(edge_index, labels):
    """
    Compute edge homophily (fraction of edges connecting same-label nodes).
    
    Args:
        edge_index: Edge indices (2 × E)
        labels: Node labels (N,)
    
    Returns:
        homophily: Edge homophily ratio
    """
    row, col = edge_index
    same_label = (labels[row] == labels[col]).sum()
    total_edges = len(row)
    return same_label / total_edges if total_edges > 0 else 0.0
