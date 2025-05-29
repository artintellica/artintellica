import numpy as np

# Create a 3x3 matrix representing 3 samples with 3 features
# Example: features could be height, weight, age
A = np.array(
    [[170, 70, 30], [165, 65, 25], [180, 80, 35]]  # Sample 1  # Sample 2  # Sample 3
)

# Compute Gram matrix for the features (columns)
gram_matrix = A.T @ A

# Compute determinant of Gram matrix
det_gram = np.linalg.det(gram_matrix)

# Check if determinant is close to zero (within numerical tolerance)
is_independent = not np.isclose(det_gram, 0, atol=1e-10)

# Print results
print("Matrix A (3 samples, 3 features):\n", A)
print("\nGram matrix (3x3):\n", gram_matrix)
print("\nDeterminant of Gram matrix:", det_gram)
print("\nAre features linearly independent?", is_independent)
