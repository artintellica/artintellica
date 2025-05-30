import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a 5x5 matrix with random integers between -5 and 5
A = np.random.randint(low=-5, high=6, size=(5, 5))

# Compute SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)

# Construct rank-2 approximation
k = 2  # Rank for approximation
A_rank2 = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Compute Frobenius norm of the difference for comparison
diff_norm = np.linalg.norm(A - A_rank2, "fro")

# Print results
print("Original matrix A (5x5):\n", A)
print("\nSingular values:", S)
print("\nRank-2 approximation A_rank2:\n", A_rank2)
print("\nFrobenius norm of difference (||A - A_rank2||_F):", diff_norm)


# Determine global min and max for consistent color scaling
vmin = min(A.min(), A_rank2.min())
vmax = max(A.max(), A_rank2.max())

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(A, cmap="viridis", vmin=vmin, vmax=vmax)
plt.title("Original Matrix A")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(A_rank2, cmap="viridis", vmin=vmin, vmax=vmax)
plt.title("Rank-2 Approximation")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(A - A_rank2, cmap="viridis")
plt.title("Difference (A - A_rank2)")
plt.colorbar()

plt.tight_layout()
plt.show()
