import numpy as np
import torch

# Define matrices A and B (from Exercise 3)
A_np = np.array([[1, 2], [3, 4]])
B_np = np.array([[0, 1], [1, 0]])

# Convert to PyTorch tensors
A_torch = torch.tensor(A_np, dtype=torch.float32)
B_torch = torch.tensor(B_np, dtype=torch.float32)

# Compute AB using PyTorch
AB_torch = A_torch @ B_torch

# Compute AB using NumPy for verification
AB_np = A_np @ B_np

# Print results
print("Matrix A (NumPy):\n", A_np)
print("\nMatrix B (NumPy):\n", B_np)
print("\nAB (PyTorch):\n", AB_torch.numpy())
print("\nAB (NumPy):\n", AB_np)

# Verify the results match
match = np.allclose(AB_torch.numpy(), AB_np)
print("\nDo PyTorch and NumPy results match?", match)
