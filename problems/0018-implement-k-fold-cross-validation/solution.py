import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # np.random.seed(42)
    if shuffle:
        idxs = np.random.permutation(n_samples)
    else:
        idxs = np.arange(n_samples)
    step_size = n_samples//k
    ans = []
    for i in range(k):
        test_idx = list(idxs[step_size*i: step_size*(i+1)])
        pre = list(idxs[:step_size*(i)])
        post = list(idxs[step_size*(i+1):])
        train_idx =  pre + post
                    
        ans.append((train_idx, test_idx))
    return ans
    