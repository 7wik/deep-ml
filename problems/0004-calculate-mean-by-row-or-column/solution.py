def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	import numpy as np

	array = np.array(matrix)
	if mode == "column":
		ax = 0
	else:
		ax = 1
	return array.mean(axis=ax).tolist()