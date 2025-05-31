# Exercise 5: Noise Filtering with SVD

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
# Original 5x5 matrix
A = np.random.rand(5, 5)

# Add random noise
noise = 0.3 * np.random.randn(5, 5)
A_noisy = A + noise

# SVD of noisy matrix
U, S, Vt = np.linalg.svd(A_noisy, full_matrices=False)

# Rank-3 approximation (noise filtering)
k = 3
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]
A_denoised = U_k @ S_k @ Vt_k

# Compare with original
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(A, cmap="viridis")
plt.title("Original Matrix")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(A_noisy, cmap="viridis")
plt.title("Noisy Matrix")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(A_denoised, cmap="viridis")
plt.title("Denoised (Rank-3)")
plt.colorbar()

plt.tight_layout()
plt.show()
