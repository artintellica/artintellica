import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a 3x3 matrix with random integers between -5 and 5
A = np.random.randint(low=-5, high=6, size=(3, 3))

# Compute eigenvalues
eigenvalues = np.linalg.eigvals(A)

# Print results
print("Matrix A (3x3):\n", A)
print("\nEigenvalues:", eigenvalues)
