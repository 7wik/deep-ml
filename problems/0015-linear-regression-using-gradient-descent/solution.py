import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    # y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros(n)  # Initialize weights to zeros
    constant = 1.0/m
    for it in range(iterations):
        diff = ((theta @ X.T) - y)
        theta = theta - alpha * (diff.reshape((-1,1)) * X).mean(axis=0)

    return theta.flatten()