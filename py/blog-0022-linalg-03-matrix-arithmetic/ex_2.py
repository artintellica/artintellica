import numpy as np
import matplotlib.pyplot as plt

# Define a 2x3 matrix
A = np.array([[1, 2, 3], [4, 5, 6]])

# Define the scalar
k = 1.5

# Scale the matrix
scaled_A = k * A

# Print results
print("Original matrix A:\n", A)
print("\nScalar k:", k)
print("\nScaled matrix (k * A):\n", scaled_A)

# Determine global min and max for consistent color scaling
vmin = min(A.min(), scaled_A.min())
vmax = max(A.max(), scaled_A.max())

# Visualize matrices as heatmaps with consistent color scale
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(A, cmap="viridis", vmin=vmin, vmax=vmax)
plt.title("Matrix A")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(scaled_A, cmap="viridis", vmin=vmin, vmax=vmax)
plt.title(f"k * A (k = {k})")
plt.colorbar()

plt.tight_layout()
plt.show()
