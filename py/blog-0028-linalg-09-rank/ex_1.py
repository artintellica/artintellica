import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a 4x4 matrix with random integers between -5 and 5
A = np.random.randint(low=-5, high=6, size=(4, 4))

# Compute rank
rank = np.linalg.matrix_rank(A)

# Print results
print("Matrix A (4x4):\n", A)
print("\nRank of A:", rank)
