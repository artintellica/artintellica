import numpy as np
import torch

# Vector from Exercise 1 (using same random seed for consistency)
np.random.seed(42)
v_np = np.random.randint(low=-5, high=6, size=3)

# Convert to PyTorch tensor
v_torch = torch.tensor(v_np, dtype=torch.float32)

# Compute NumPy norms
l1_norm_np = np.sum(np.abs(v_np))  # L1 norm
l2_norm_np = np.linalg.norm(v_np)  # L2 norm
linf_norm_np = np.max(np.abs(v_np))  # L∞ norm

# Compute PyTorch norms
l1_norm_torch = torch.sum(torch.abs(v_torch))  # L1 norm
l2_norm_torch = torch.norm(v_torch)  # L2 norm
linf_norm_torch = torch.max(torch.abs(v_torch))  # L∞ norm

# Print results
print("Vector v (NumPy):", v_np)
print("\nNumPy norms:")
print("L1 norm:", l1_norm_np)
print("L2 norm:", l2_norm_np)
print("L∞ norm:", linf_norm_np)
print("\nPyTorch norms:")
print("L1 norm:", l1_norm_torch.item())
print("L2 norm:", l2_norm_torch.item())
print("L∞ norm:", linf_norm_torch.item())

# Verify results match
print("\nDo NumPy and PyTorch results match?")
print("L1 norm match:", np.isclose(l1_norm_np, l1_norm_torch.item()))
print("L2 norm match:", np.isclose(l2_norm_np, l2_norm_torch.item()))
print("L∞ norm match:", np.isclose(linf_norm_np, linf_norm_torch.item()))
