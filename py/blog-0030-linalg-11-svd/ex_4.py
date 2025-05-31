# Exercise 4: Image Compression with SVD

import numpy as np
import matplotlib.pyplot as plt

# Option 1: Load a grayscale image (uncomment and provide a path)
# from matplotlib.image import imread
# img = imread('path_to_grayscale_image.png')
# if img.ndim == 3:
#     img = img.mean(axis=2)  # Convert to grayscale if needed

# Option 2: Create a small random matrix as a "fake image"
np.random.seed(0)
img = np.random.rand(50, 50)

# Compute SVD
U, S, Vt = np.linalg.svd(img, full_matrices=False)

# Rank-10 approximation
k = 10
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]
img_rank10 = U_k @ S_k @ Vt_k

# Visualization
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_rank10, cmap="gray")
plt.title("Rank-10 Approximation")
plt.axis("off")

plt.tight_layout()
plt.show()
