import numpy as np

# Define vectors
u = np.array([1, 2])  # Vector to project
v = np.array([3, 1])  # Vector to project onto

# Compute projection of u onto v
dot_uv = np.dot(u, v)  # Dot product u · v
norm_v_squared = np.sum(v**2)  # Squared length of v (v · v)
projection = (dot_uv / norm_v_squared) * v  # Projection formula

# Compute orthogonal error vector
error = u - projection

# Verify orthogonality by computing dot product of error and v
error_dot_v = np.dot(error, v)

# Print results
print("Vector u:", u)
print("Vector v:", v)
print("Projection of u onto v:", projection)
print("Orthogonal error (u - proj_v(u)):", error)
print("Dot product of error and v (should be ~0):", error_dot_v)
print("Is error orthogonal to v?", np.isclose(error_dot_v, 0, atol=1e-10))
