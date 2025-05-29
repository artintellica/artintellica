import numpy as np
import torch

# Vectors from Exercise 1 (using same random seed for consistency)
np.random.seed(42)
u_np = np.random.randint(low=-5, high=6, size=3)
v_np = np.random.randint(low=-5, high=6, size=3)

# Convert to PyTorch tensors
u_torch = torch.tensor(u_np, dtype=torch.float32)
v_torch = torch.tensor(v_np, dtype=torch.float32)

# Compute cosine similarity using PyTorch
cosine_sim_torch = torch.cosine_similarity(u_torch, v_torch, dim=0)

# Compute cosine similarity using NumPy for verification
dot_product = np.dot(u_np, v_np)
norm_u = np.linalg.norm(u_np)
norm_v = np.linalg.norm(v_np)
cosine_sim_np = dot_product / (norm_u * norm_v) if norm_u * norm_v != 0 else 0

# Print results
print("Vector u (NumPy):", u_np)
print("Vector v (NumPy):", v_np)
print("Cosine similarity (PyTorch):", cosine_sim_torch.item())
print("Cosine similarity (NumPy):", cosine_sim_np)
print(
    "Do PyTorch and NumPy results match?",
    np.allclose(cosine_sim_torch.item(), cosine_sim_np),
)
