import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # # Your code here
    # pass
    B = A.T @ A
    a, b, d = B[0,0], B[0,1], B[1,1]
    theta = 0.5*np.arctan2(2*b, (d-a))
    c, s = np.cos(theta), np.sin(theta)
    V = np.array([[c, s], [-s, c]])
    lam = np.diag(V.T @ B @ V)
    idx = np.argsort(lam)[::-1]
    lam = lam[idx]
    V = V[:, idx]
    S = np.sqrt(lam)
    U = np.zeros((2,2))
    if S[0] > 1e-12:
        U[:,0] = A @ V[:, 0]/S[0]
    else:
        U[:,0] = np.array([1.0, 0.0])
    
    if S[1] > 1e-12:
        U[:, 1] = A @ V[:, 1]/S[1]
    else:
        U[:,1] = np.array([0.0, 1.0])
    return U, S, V.T
    