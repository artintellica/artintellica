import numpy as np
import matplotlib.pyplot as plt

# Create a 4x3 matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# Compute SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)

# Rank-1 approximation
k = 1
U_k = U[:, :k]
Sigma_k = np.diag(S[:k])
Vt_k = Vt[:k, :]
A_rank1 = U_k @ Sigma_k @ Vt_k

# Compute Frobenius norm of difference
diff_norm = np.linalg.norm(A - A_rank1, "fro")

# Print results
print("Rank-1 approximation A_rank1:\n", A_rank1)
print("\nFrobenius norm of difference (||A - A_rank1||_F):", diff_norm)


# Consistent color scale
vmin = min(A.min(), A_rank1.min())
vmax = max(A.max(), A_rank1.max())

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(A, cmap="viridis", vmin=vmin, vmax=vmax)
plt.title("Original A")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(A_rank1, cmap="viridis", vmin=vmin, vmax=vmax)
plt.title("Rank-1 Approximation")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(A - A_rank1, cmap="plasma")
plt.title("Difference")
plt.colorbar()
plt.tight_layout()
plt.show()
