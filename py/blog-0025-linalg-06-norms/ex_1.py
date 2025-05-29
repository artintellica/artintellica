import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a 3D vector with random integers between -5 and 5
v = np.random.randint(low=-5, high=6, size=3)

# Compute norms
l1_norm = np.sum(np.abs(v))  # L1 norm
l2_norm = np.linalg.norm(v)  # L2 norm
linf_norm = np.max(np.abs(v))  # L∞ norm

# Print results
print("Vector v:", v)
print("L1 norm:", l1_norm)
print("L2 norm:", l2_norm)
print("L∞ norm:", linf_norm)
