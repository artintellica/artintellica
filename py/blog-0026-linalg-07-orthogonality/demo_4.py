import torch

# Convert to PyTorch tensors
u_torch = torch.tensor([1.0, 2.0])
v_torch = torch.tensor([3.0, 1.0])

# Compute projection
dot_uv_torch = torch.dot(u_torch, v_torch)
norm_v_squared_torch = torch.sum(v_torch**2)
projection_torch = (dot_uv_torch / norm_v_squared_torch) * v_torch

# Print result
print("PyTorch projection of u onto v:", projection_torch.numpy())
