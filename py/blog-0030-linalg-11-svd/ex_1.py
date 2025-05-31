import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a 3x4 matrix with random integers between -5 and 5
A = np.random.randint(low=-5, high=6, size=(3, 4))

# Compute SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)

# Construct Sigma as a 3x4 diagonal matrix
Sigma = np.zeros((A.shape[0], A.shape[1]))
np.fill_diagonal(Sigma, S)

# Reconstruct the matrix
A_reconstructed = U @ Sigma @ Vt

# Verify reconstruction
reconstruction_matches = np.allclose(A, A_reconstructed, atol=1e-10)

# Print results
print("Original matrix A (3x4):\n", A)
print("\nSingular values:", S)
print("\nU (3x3):\n", U)
print("\nSigma (3x4):\n", Sigma)
print("\nVt (4x4):\n", Vt)
print("\nReconstructed matrix A:\n", A_reconstructed)
print("\nReconstruction matches original?", reconstruction_matches)
