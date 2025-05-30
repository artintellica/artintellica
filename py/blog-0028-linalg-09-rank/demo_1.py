import numpy as np

# Define a 4x3 matrix
A = np.array([[1, 2, 3], [2, 4, 6], [3, 1, 4], [0, 0, 1]])

# Compute rank
rank = np.linalg.matrix_rank(A)

# Print results
print("Matrix A (4x3):\n", A)
print("\nRank of A:", rank)
