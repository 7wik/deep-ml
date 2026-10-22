import numpy as np

def batchDG(X, y, weights, learning_rate, n_epochs):
    constant = 2.0/X.shape[0]
    for epoch in range(n_epochs):
        residual = (X @ weights - y)
        gradients = residual @ X
        weights = weights - constant*learning_rate*gradients
    return weights

def minibatchGD(X, y, weights, learning_rate, n_epochs, batch_size):
    constant = 2.0/batch_size
    for epoch in range(n_epochs):
        for i in range(X.shape[0]//batch_size):
            batch = X[i*batch_size : (i+1)*batch_size]
            batch_y = y[i*batch_size : (i+1)*batch_size]
            residual = (batch @ weights - batch_y)
            gradients = residual @ batch
            weights = weights - constant*learning_rate*gradients
    return weights

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    if method == 'batch':
        return batchDG(X, y, weights, learning_rate, n_epochs)
    elif method == 'mini_batch':
        return minibatchGD(X, y, weights, learning_rate, n_epochs, batch_size)
    return minibatchGD(X, y, weights, learning_rate, n_epochs, 1)



