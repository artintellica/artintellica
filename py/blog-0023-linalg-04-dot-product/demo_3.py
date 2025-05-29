import numpy as np
import torch

# Define two 2D vectors
u = np.array([1, 2])
v = np.array([3, 1])

# Compute dot product
dot_product = np.dot(u, v)

# Compute norms
norm_u = np.linalg.norm(u)
norm_v = np.linalg.norm(v)


# Convert to PyTorch tensors
u_torch = torch.tensor(u, dtype=torch.float32)
v_torch = torch.tensor(v, dtype=torch.float32)

# Compute cosine similarity
cosine_sim_torch = torch.cosine_similarity(u_torch, v_torch, dim=0)

# Print result
print("PyTorch cosine similarity:", cosine_sim_torch.item())
