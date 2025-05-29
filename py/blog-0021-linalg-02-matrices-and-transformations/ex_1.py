import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a 3x4 matrix with random integers between 0 and 9
matrix = np.random.randint(low=0, high=10, size=(3, 4))

# Compute the transpose
matrix_transpose = matrix.T

# Print the matrix and its transpose
print("Original 3x4 matrix:\n", matrix)
print("\nTranspose (4x3 matrix):\n", matrix_transpose)
