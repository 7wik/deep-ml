# --- cell 1 ---
import numpy as np

def compute_softmax(X):
    X = X - X.max(axis=-1, keepdims=True)
    num = np.exp(X)
    den = num.sum(axis=-1, keepdims=True)
    return num/den

def attention(Q, K, V):
    """
    Compute attention over a sequence.
    
    Args:
        Q: Query matrix, shape (batch_size, query_len, dim)
        K: Key matrix, shape (batch_size, key_len, dim)
        V: Value matrix, shape (batch_size, key_len, dim)
    
    Returns:
        output: Attended values, shape (batch_size, query_len, dim)
    
    The attention mechanism should:
    1. Compute compatibility between queries and keys
    2. Convert to attention weights (non-negative, sum to 1)
    3. Use weights to compute weighted sum of values
    """
    batch_size, query_len, dim = Q.shape
    _, key_len, _ = K.shape
    scale = 1.0/np.sqrt(dim)
    scores = compute_softmax(scale*Q@K.transpose((0, 2, 1)))
    return scores@V

# --- cell 2 ---
# import dml_data
# X, y = dml_data.load_xy("imdb-reviews")
#

# --- cell 3 ---
