import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a 2x2 matrix with random integers between -5 and 5
A = np.random.randint(low=-5, high=6, size=(2, 2))

# Compute eigenvalues
eigenvalues = np.linalg.eigvals(A)

# Compute absolute values of eigenvalues
abs_eigenvalues = np.abs(eigenvalues)

# Check stability (all |eigenvalues| < 1)
is_stable = np.all(abs_eigenvalues < 1)

# Print results
print("Matrix A (2x2):\n", A)
print("\nEigenvalues:", eigenvalues)
print("\nAbsolute values of eigenvalues:", abs_eigenvalues)
print("\nIs the system stable? (All |eigenvalues| < 1):", is_stable)
