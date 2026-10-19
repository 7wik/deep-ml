import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	# return []
	N = len(vectors[0])
	X = np.array(vectors) # dxN
	mu = np.mean(X, axis=1, keepdims=True) # dx1
	Xc = X - mu # dxN
	cov = (Xc @ X.T)/(N-1)
	return cov


	
