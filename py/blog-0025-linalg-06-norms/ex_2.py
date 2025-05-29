import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Create two 2D vectors with random integers between -5 and 5
u = np.random.randint(low=-5, high=6, size=2)
v = np.random.randint(low=-5, high=6, size=2)

# Compute distances
l1_dist = np.sum(np.abs(u - v))  # L1 distance
l2_dist = np.linalg.norm(u - v)  # L2 distance
linf_dist = np.max(np.abs(u - v))  # L∞ distance

# Print results
print("Vector u:", u)
print("Vector v:", v)
print("L1 distance:", l1_dist)
print("L2 distance:", l2_dist)
print("L∞ distance:", linf_dist)

# Visualize vectors and L2 distance
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, u[0], u[1], color="red", scale=1, scale_units="xy", angles="xy")
plt.quiver(0, 0, v[0], v[1], color="blue", scale=1, scale_units="xy", angles="xy")
plt.plot([u[0], v[0]], [u[1], v[1]], "g--", label="L2 distance")
plt.text(u[0], u[1], "u", color="red", fontsize=12)
plt.text(v[0], v[1], "v", color="blue", fontsize=12)
plt.grid(True)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Vectors and L2 Distance")
plt.legend()
plt.show()
