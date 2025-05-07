import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

def reduced_rank_regression(X, Y, ranks=None):
    """
    Perform reduced-rank regression to find communication subspace.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Source area activity (trials × neurons)
    Y : numpy.ndarray
        Target area activity (trials × neurons)
    ranks : list
        List of ranks to try (if None, tries all possible ranks)
        
    Returns:
    --------
    W : dict
        Regression weights for each rank
    explained_var : dict
        Explained variance for each rank
    """
    if ranks is None:
        ranks = range(1, min(X.shape[1], Y.shape[1]) + 1)
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # Compute the OLS solution
    X_pinv = np.linalg.pinv(X_centered)
    B_ols = X_pinv @ Y_centered
    
    # Compute SVD of the OLS solution
    U, s, Vt = np.linalg.svd(B_ols, full_matrices=False)
    
    # Store weights and explained variance for each rank
    W = {}
    explained_var = {}
    
    for rank in ranks:
        # Truncate SVD components to the desired rank
        B_rrr = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
        W[rank] = B_rrr
        
        # Compute explained variance
        Y_pred = X_centered @ B_rrr
        explained_var[rank] = 1 - np.sum((Y_centered - Y_pred)**2) / np.sum(Y_centered**2)
    
    return W, explained_var

def cross_validate_rrr(X, Y, ranks=None, n_splits=5):
    """
    Cross-validate RRR to find optimal rank.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Source area activity (trials × neurons)
    Y : numpy.ndarray
        Target area activity (trials × neurons)
    ranks : list
        List of ranks to try (if None, tries all possible ranks)
    n_splits : int
        Number of cross-validation folds
        
    Returns:
    --------
    avg_scores : dict
        Average cross-validated R² for each rank
    """
    if ranks is None:
        ranks = range(1, min(X.shape[1], Y.shape[1]) + 1)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = {rank: [] for rank in ranks}
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        # Train RRR on training data
        W, _ = reduced_rank_regression(X_train, Y_train, ranks)
        
        # Evaluate on test data
        X_test_centered = X_test - np.mean(X_train, axis=0)
        Y_test_centered = Y_test - np.mean(Y_train, axis=0)
        
        for rank in ranks:
            Y_pred = X_test_centered @ W[rank]
            score = 1 - np.sum((Y_test_centered - Y_pred)**2) / np.sum(Y_test_centered**2)
            cv_scores[rank].append(score)
    
    # Average scores across folds
    avg_scores = {rank: np.mean(scores) for rank, scores in cv_scores.items()}
    
    return avg_scores

def get_communication_subspace(X, Y, optimal_rank):
    """
    Get the communication subspace at the optimal rank.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Source area activity (trials × neurons)
    Y : numpy.ndarray
        Target area activity (trials × neurons)
    optimal_rank : int
        Optimal rank determined by cross-validation
        
    Returns:
    --------
    communication_dimensions : numpy.ndarray
        The communication dimensions (patterns)
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # Compute the OLS solution
    X_pinv = np.linalg.pinv(X_centered)
    B_ols = X_pinv @ Y_centered
    
    # Compute SVD of the OLS solution
    U, s, Vt = np.linalg.svd(B_ols, full_matrices=False)
    
    # Extract the communication dimensions
    communication_dimensions = U[:, :optimal_rank]
    
    return communication_dimensions

def subspace_alignment(basis1, basis2):
    """
    Measure alignment between two subspaces.
    
    Parameters:
    -----------
    basis1 : numpy.ndarray
        First subspace basis (orthonormal)
    basis2 : numpy.ndarray
        Second subspace basis (orthonormal)
        
    Returns:
    --------
    alignment : float
        Alignment between subspaces (0-1, where 1 means perfectly aligned)
    """
    # Make sure bases are orthonormal
    basis1_ortho, _ = np.linalg.qr(basis1)
    basis2_ortho, _ = np.linalg.qr(basis2)
    
    # Compute alignment
    alignment = np.linalg.norm(basis1_ortho.T @ basis2_ortho, ord='fro')**2 / min(basis1.shape[1], basis2.shape[1])
    
    return alignment