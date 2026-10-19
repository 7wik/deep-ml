import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	# return transformed_matrix
	try:
		T_inv = np.linalg.inv(T)
	except np.linalg.LinAlgError:
		return -1

	try:
		S_inv = np.linalg.inv(S)
	except np.linalg.LinAlgError:
		return -1
	return np.dot(np.dot(T_inv, A), S)

	