import numpy as np
import torch

# Define a 4x3 matrix
A = np.array([[1, 2, 3], [2, 4, 6], [3, 1, 4], [0, 0, 1]])

tol = 1e-10

# Convert to PyTorch tensor
A_torch = torch.tensor(A, dtype=torch.float32)

# Compute rank via SVD
_, S_torch, _ = torch.svd(A_torch)
rank_torch = torch.sum(S_torch > tol).item()

print("PyTorch rank:", rank_torch)
