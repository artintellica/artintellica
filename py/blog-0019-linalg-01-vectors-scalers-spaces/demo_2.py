import numpy as np
import torch

# Define two vectors
u = np.array([1, 2])
v = np.array([2, -1])

# Addition
sum_uv = u + v
print("u + v =", sum_uv)

# Scaling
scaled_v = 2 * v
print("2 * v =", scaled_v)

# Now in PyTorch
u_torch = torch.tensor([1, 2], dtype=torch.float32)
v_torch = torch.tensor([2, -1], dtype=torch.float32)

# Addition in PyTorch
sum_uv_torch = u_torch + v_torch
print("u + v (PyTorch) =", sum_uv_torch)

# Scaling in PyTorch
scaled_v_torch = 2 * v_torch
print("2 * v (PyTorch) =", scaled_v_torch)
