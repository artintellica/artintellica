import numpy as np

# Define two sets of 2D vectors
independent_vectors = np.array([[1, 0], [0, 1]])  # Linearly independent
dependent_vectors = np.array([[1, 2], [2, 4]])  # Linearly dependent

# Compute Gram matrices
gram_independent = independent_vectors.T @ independent_vectors
gram_dependent = dependent_vectors.T @ dependent_vectors

# Check determinants
det_independent = np.linalg.det(gram_independent)
det_dependent = np.linalg.det(gram_dependent)

# Print results
print("Independent vectors:\n", independent_vectors)
print("Gram matrix (independent):\n", gram_independent)
print("Determinant (independent):", det_independent)
print("\nDependent vectors:\n", dependent_vectors)
print("Gram matrix (dependent):\n", gram_dependent)
print("Determinant (dependent):", det_dependent)
