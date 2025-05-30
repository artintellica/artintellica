import numpy as np

# Set random seed for reproducibility (same as Exercise 1)
np.random.seed(42)

# Create the same 4x4 matrix from Exercise 1
A = np.random.randint(low=-5, high=6, size=(4, 4))

# Compute SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)

# Tolerance for identifying zero singular values
tol = 1e-10

# Nullspace basis (columns of V corresponding to zero singular values)
nullspace_basis = Vt.T[:, S < tol]

# Compute rank and nullity
rank = np.sum(S > tol)
nullity = A.shape[1] - rank

# Print results
print("Matrix A (4x4):\n", A)
print("\nSingular values:", S)
print("\nRank of A:", rank)
print(
    "\nNullspace basis:\n",
    nullspace_basis if nullspace_basis.size > 0 else "Empty (full rank)",
)
print("\nDimension of nullspace (nullity):", nullity)
