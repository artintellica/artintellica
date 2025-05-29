import numpy as np

# Define two 2D vectors
u = np.array([1, 2])
v = np.array([3, 1])

# Compute dot product
dot_product = np.dot(u, v)

# Compute norms
norm_u = np.linalg.norm(u)
norm_v = np.linalg.norm(v)

# Compute cosine similarity
cosine_sim = dot_product / (norm_u * norm_v)

# Print results
print("Norm of u:", norm_u)
print("Norm of v:", norm_v)
print("Cosine similarity:", cosine_sim)
