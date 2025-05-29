import numpy as np
import torch

# Define a vector
v = np.array([3, 4])

# Define another vector
u = np.array([1, 1])


# Convert to PyTorch tensors
u_torch = torch.tensor(u, dtype=torch.float32)
v_torch = torch.tensor(v, dtype=torch.float32)

# Compute norms for v
l1_norm_torch = torch.sum(torch.abs(v_torch))
l2_norm_torch = torch.norm(v_torch)
linf_norm_torch = torch.max(torch.abs(v_torch))

# Compute distances
l1_dist_torch = torch.sum(torch.abs(u_torch - v_torch))
l2_dist_torch = torch.norm(u_torch - v_torch)
linf_dist_torch = torch.max(torch.abs(u_torch - v_torch))

# Print results
print("PyTorch L1 norm (v):", l1_norm_torch.item())
print("PyTorch L2 norm (v):", l2_norm_torch.item())
print("PyTorch L∞ norm (v):", linf_norm_torch.item())
print("PyTorch L1 distance:", l1_dist_torch.item())
print("PyTorch L2 distance:", l2_dist_torch.item())
print("PyTorch L∞ distance:", linf_dist_torch.item())
