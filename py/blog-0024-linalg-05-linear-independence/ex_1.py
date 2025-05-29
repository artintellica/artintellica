import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create three 2D vectors with random integers between -5 and 5
v1 = np.random.randint(low=-5, high=6, size=2)
v2 = np.random.randint(low=-5, high=6, size=2)
v3 = np.random.randint(low=-5, high=6, size=2)

# Combine vectors into a matrix (2x3)
vectors = np.array([v1, v2, v3]).T  # Transpose to get 2x3 matrix

# Compute Gram matrix (3x3)
gram_matrix = vectors.T @ vectors

# Compute determinant
det_gram = np.linalg.det(gram_matrix)

# Print results
print("Vector v1:", v1)
print("Vector v2:", v2)
print("Vector v3:", v3)
print("\nVectors as matrix (2x3):\n", vectors)
print("\nGram matrix (3x3):\n", gram_matrix)
print("\nDeterminant of Gram matrix:", det_gram)
print("\nLinearly independent?", det_gram != 0)

# Note about three 2D vectors
print(
    "\nNote: Three 2D vectors are always linearly dependent in R^2, as the dimension is 2."
)
