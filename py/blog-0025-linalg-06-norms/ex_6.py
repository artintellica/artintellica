import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate 10 random 2D vectors with integers between -5 and 5
vectors = np.random.randint(low=-5, high=6, size=(10, 2))

# Compute L1 and L2 norms for each vector
l1_norms = np.sum(np.abs(vectors), axis=1)  # L1 norm (sum of absolute values)
l2_norms = np.linalg.norm(vectors, axis=1)  # L2 norm (Euclidean norm)

# Print results
print("Vectors (10x2):\n", vectors)
print("\nL1 norms:", l1_norms)
print("\nL2 norms:", l2_norms)

# Create scatter plot of L1 vs. L2 norms
plt.figure(figsize=(6, 6))
plt.scatter(l1_norms, l2_norms, color="blue", s=50)
plt.xlabel("L1 Norm")
plt.ylabel("L2 Norm")
plt.title("L1 vs. L2 Norms for 10 Random 2D Vectors")
plt.grid(True)
for i, (x, y) in enumerate(zip(l1_norms, l2_norms)):
    plt.text(x, y, f"v{i}", fontsize=10, ha="right")
plt.show()
