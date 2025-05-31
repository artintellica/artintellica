import numpy as np

# Use the matrix from Exercise 1 (replace this with your actual matrix if needed)
np.random.seed(0)
A = np.random.randint(-5, 6, size=(3, 4))

# Compute SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Rank-2 approximation
k = 2
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]
A_rank2 = U_k @ S_k @ Vt_k

# Frobenius norm of the difference
diff_norm = np.linalg.norm(A - A_rank2, 'fro')

print("Original matrix A:\n", A)
print("\nRank-2 approximation A_rank2:\n", A_rank2)
print("\nFrobenius norm of difference (||A - A_rank2||_F):", diff_norm)
