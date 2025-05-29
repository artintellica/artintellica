import numpy as np

# Create a 2D vector
v = np.array([4, 3])

# Compute original norms
l1_norm = np.sum(np.abs(v))  # L1 norm
l2_norm = np.linalg.norm(v)  # L2 norm

# Scale the vector by 0.5
scaled_v = 0.5 * v

# Compute norms of scaled vector
l1_norm_scaled = np.sum(np.abs(scaled_v))  # L1 norm
l2_norm_scaled = np.linalg.norm(scaled_v)  # L2 norm

# Print results
print("Original vector v:", v)
print("L1 norm (original):", l1_norm)
print("L2 norm (original):", l2_norm)
print("\nScaled vector (0.5 * v):", scaled_v)
print("L1 norm (scaled):", l1_norm_scaled)
print("L2 norm (scaled):", l2_norm_scaled)
