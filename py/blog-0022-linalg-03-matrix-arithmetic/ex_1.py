import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Create two 3x3 matrices with random integers between 0 and 9
A = np.random.randint(low=0, high=10, size=(3, 3))
B = np.random.randint(low=0, high=10, size=(3, 3))

# Compute their sum
C = A + B

# Print results
print("Matrix A:\n", A)
print("\nMatrix B:\n", B)
print("\nA + B:\n", C)

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
