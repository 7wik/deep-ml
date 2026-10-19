def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	import numpy as np
	return np.linalg.eigvals(np.array(matrix)).real.tolist()