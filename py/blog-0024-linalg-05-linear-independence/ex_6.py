import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a 4x3 matrix with random integers between -5 and 5
A = np.random.randint(low=-5, high=6, size=(4, 3))

# Compute the rank of the matrix
rank = np.linalg.matrix_rank(A)

# Determine if columns are linearly independent
num_columns = A.shape[1]
are_independent = rank == num_columns

# Print results
print("Matrix A (4x3):\n", A)
print("\nRank of matrix A:", rank)
print("Number of columns:", num_columns)
print("Are columns linearly independent?", are_independent)
print("\nNote: Columns are linearly independent if rank equals the number of columns.")
