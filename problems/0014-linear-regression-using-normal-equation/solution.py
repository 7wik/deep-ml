import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X = np.array(X)
	y = np.array(y)
	AtA = X.T @ X
	AtA_inv = np.linalg.inv(AtA)
	AtY = X.T @ y
	return AtA_inv @ AtY