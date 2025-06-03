import numpy as np

# Create a symmetric matrix
A = np.array([[4, 1], [1, 3]])

# Check if symmetric
is_symmetric = np.allclose(A, A.T)
print("Is symmetric:", is_symmetric)

# Check eigenvalues (all should be positive for positive definite)
eigenvalues = np.linalg.eigvals(A)
print("Eigenvalues:", eigenvalues)
is_positive_definite = np.all(eigenvalues > 0)
print("Is positive definite (eigenvalue test):", is_positive_definite)

# Cholesky decomposition (only works for positive definite matrices)
try:
    L = np.linalg.cholesky(A)
    print("Cholesky decomposition (L):")
    print(L)
    print("Reconstructed A from L L^T:")
    print(L @ L.T)
except np.linalg.LinAlgError:
    print("Matrix is not positive definite; Cholesky decomposition failed.")
