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

# Define a 3x3 matrix with linearly dependent columns
A_dep = np.array([[1, 2, 4], [2, 1, 5], [3, 0, 6]])  # Third column = 2 * first + second

# Compute rank and SVD
rank_dep = np.linalg.matrix_rank(A_dep)
U_dep, S_dep, Vt_dep = np.linalg.svd(A_dep, full_matrices=True)
nullspace_basis_dep = Vt_dep.T[:, S_dep < tol]

# Print results
print("Matrix A_dep (3x3):\n", A_dep)
print("\nRank of A_dep:", rank_dep)
print("Singular values:", S_dep)
print("\nNullspace basis:\n", nullspace_basis_dep)
print("Nullity:", A_dep.shape[1] - rank_dep)
