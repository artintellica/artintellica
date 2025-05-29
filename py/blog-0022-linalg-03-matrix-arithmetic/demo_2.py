import numpy as np
import matplotlib.pyplot as plt

# Define two 2x2 matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Scale matrix A by 2
k = 2
scaled_A = k * A

# Print result
print("Scalar k:", k)
print("Scaled matrix (k * A):\n", scaled_A)

vmin = min(A.min(), B.min())
vmax = max(A.max(), B.max())

# Visualize
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(A, cmap="viridis", vmin=vmin, vmax=vmax)
plt.title("Matrix A")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(scaled_A, cmap="viridis", vmin=vmin, vmax=vmax)
plt.title("k * A")
plt.colorbar()
plt.tight_layout()
plt.show()
