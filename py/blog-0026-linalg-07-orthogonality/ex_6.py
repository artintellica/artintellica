import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a 3x3 matrix with random integers between -5 and 5
A = np.random.randint(low=-5, high=6, size=(3, 3))


# Gram-Schmidt process on columns
def gram_schmidt_columns(A):
    n = A.shape[1]
    U = np.zeros_like(A, dtype=float)  # Store orthonormal vectors

    for i in range(n):
        # Start with the i-th column
        w = A[:, i].astype(float)

        # Subtract projections onto previous orthonormal vectors
        for j in range(i):
            w -= np.dot(w, U[:, j]) * U[:, j]

        # Normalize to get orthonormal vector
        U[:, i] = w / np.linalg.norm(w)

    return U


# Apply Gram-Schmidt
U = gram_schmidt_columns(A)

# Verify orthonormality
# Compute dot products for all pairs
dot_products = U.T @ U  # Should be identity matrix for orthonormal set
norms = np.array([np.linalg.norm(U[:, i]) for i in range(3)])  # Should be 1

# Check if orthonormal
is_orthogonal = np.allclose(dot_products, np.eye(3), atol=1e-10)
is_unit_norm = np.allclose(norms, 1, atol=1e-10)
is_orthonormal = is_orthogonal and is_unit_norm

# Print results
print("Original matrix A (3x3):\n", A)
print("\nOrthonormal basis U (columns):\n", U)
print("\nDot products (U^T U, should be identity):\n", dot_products)
print("Norms of columns (should be 1):", norms)
print("\nIs orthogonal?", is_orthogonal)
print("All unit norms?", is_unit_norm)
print("Is orthonormal?", is_orthonormal)
