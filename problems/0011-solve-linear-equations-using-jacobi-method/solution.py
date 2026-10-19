import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	# return x
	x = np.zeros(A.shape[0])
	for i in range(n):
		a_diag = np.diag(A)
		a_diag_inv = 1.0/a_diag
		ax = A @ x
		aii_xi = a_diag * x
		aij_xj = (ax - aii_xi)
		x = (b - aij_xj)*a_diag_inv
	return x