import numpy as np

# Define a 4x3 matrix
A = np.array([[1, 2, 3], [2, 4, 6], [3, 1, 4], [0, 0, 1]])

# Compute rank
rank = np.linalg.matrix_rank(A)

# Compute SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)

# Nullspace basis (columns of V corresponding to zero singular values)
tol = 1e-10
nullspace_basis = Vt.T[:, S < tol]

# Print results
print("Singular values:", S)
print(
    "\nNullspace basis (if any):\n",
    nullspace_basis if nullspace_basis.size > 0 else "Empty (full column rank)",
)
print("\nNullity (n - rank):", A.shape[1] - rank)
