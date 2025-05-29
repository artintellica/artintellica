import numpy as np
import matplotlib.pyplot as plt

# Define two 2x2 matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix addition
C = A + B

# Print results
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("A + B:\n", C)

# Determine global min and max for consistent color scaling
vmin = min(A.min(), B.min(), C.min())
vmax = max(A.max(), B.max(), C.max())

# Visualize matrices as heatmaps with consistent color scale
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(A, cmap="viridis", vmin=vmin, vmax=vmax)
plt.title("Matrix A")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(B, cmap="viridis", vmin=vmin, vmax=vmax)
plt.title("Matrix B")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(C, cmap="viridis", vmin=vmin, vmax=vmax)
plt.title("A + B")
plt.colorbar()

plt.tight_layout()
plt.show()
