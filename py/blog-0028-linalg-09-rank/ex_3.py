import numpy as np
import torch

# Set random seed for reproducibility (same as Exercise 1)
np.random.seed(42)

# Create the same 4x4 matrix from Exercise 1
A_np = np.random.randint(low=-5, high=6, size=(4, 4))

# Convert to PyTorch tensor
A_torch = torch.tensor(A_np, dtype=torch.float32)

# Compute rank using NumPy
rank_np = np.linalg.matrix_rank(A_np)

# Compute rank using PyTorch via SVD
_, S_torch, _ = torch.svd(A_torch)
tol = 1e-10
rank_torch = torch.sum(S_torch > tol).item()

# Verify ranks match
ranks_match = rank_np == rank_torch

# Print results
print("Matrix A (NumPy):\n", A_np)
print("\nRank (NumPy):", rank_np)
print("\nRank (PyTorch):", rank_torch)
print("\nDo NumPy and PyTorch ranks match?", ranks_match)
