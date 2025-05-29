import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create three 2D vectors with random integers between -5 and 5
v1 = np.random.randint(low=-5, high=6, size=2)
v2 = np.random.randint(low=-5, high=6, size=2)
v3 = np.random.randint(low=-5, high=6, size=2)

# Compute pairwise dot products
dot_v1_v2 = np.dot(v1, v2)
dot_v1_v3 = np.dot(v1, v3)
dot_v2_v3 = np.dot(v2, v3)

# Check orthogonality (dot product close to zero)
is_v1_v2_orthogonal = np.isclose(dot_v1_v2, 0, atol=1e-10)
is_v1_v3_orthogonal = np.isclose(dot_v1_v3, 0, atol=1e-10)
is_v2_v3_orthogonal = np.isclose(dot_v2_v3, 0, atol=1e-10)

# Print results
print("Vector v1:", v1)
print("Vector v2:", v2)
print("Vector v3:", v3)
print("\nDot product v1 · v2:", dot_v1_v2)
print("Dot product v1 · v3:", dot_v1_v3)
print("Dot product v2 · v3:", dot_v2_v3)
print("\nv1 and v2 orthogonal?", is_v1_v2_orthogonal)
print("v1 and v3 orthogonal?", is_v1_v3_orthogonal)
print("v2 and v3 orthogonal?", is_v2_v3_orthogonal)
