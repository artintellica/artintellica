import numpy as np

# Define a 2x2 matrix
A = np.array([[2, 1], [1, 2]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Print results
print("Matrix A:\n", A)
print("\nEigenvalues:", eigenvalues)
print("\nEigenvectors (columns):\n", eigenvectors)
