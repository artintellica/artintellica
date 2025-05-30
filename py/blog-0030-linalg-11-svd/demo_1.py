import numpy as np

# Create a 4x3 matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# Compute SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)

# Reconstruct Sigma as a 4x3 diagonal matrix
Sigma = np.zeros((A.shape[0], A.shape[1]))
np.fill_diagonal(Sigma, S)

# Reconstruct A
A_reconstructed = U @ Sigma @ Vt

# Print results
print("Matrix A (4x3):\n", A)
print("\nSingular values:", S)
print("\nU (4x4):\n", U)
print("\nSigma (4x3):\n", Sigma)
print("\nVt (3x3):\n", Vt)
print("\nReconstructed A:\n", A_reconstructed)
print("\nReconstruction matches original?", np.allclose(A, A_reconstructed))
