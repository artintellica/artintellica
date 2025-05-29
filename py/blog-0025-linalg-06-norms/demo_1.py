import numpy as np
import matplotlib.pyplot as plt

# Define a vector
v = np.array([3, 4])

# Compute norms
l1_norm = np.sum(np.abs(v))  # L1 norm
l2_norm = np.linalg.norm(v)  # L2 norm
linf_norm = np.max(np.abs(v))  # L∞ norm

# Print results
print("Vector v:", v)
print("L1 norm:", l1_norm)
print("L2 norm:", l2_norm)
print("L∞ norm:", linf_norm)

# Visualize vector
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, v[0], v[1], color="blue", scale=1, scale_units="xy", angles="xy")
plt.text(v[0], v[1], "v", color="blue", fontsize=12)
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Vector for Norms")
plt.show()
