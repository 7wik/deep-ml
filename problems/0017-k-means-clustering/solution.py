import numpy as np
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	centroids = np.array(initial_centroids)
	X = np.array(points)
	m, n = X.shape
	iters = max_iterations
	for i in range(iters):
		dists = np.square(centroids).sum(axis=1) - 2*X @ centroids.T
		idxs = np.argmin(dists, axis=1)
		centroids = np.array([X[idxs == j].mean(axis=0) for j in range(k)])

	
	
	return np.round(centroids, 4)