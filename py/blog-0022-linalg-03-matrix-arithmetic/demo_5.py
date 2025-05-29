import torch
import numpy as np

# Define two 2x2 matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Convert to PyTorch tensors
A_torch = torch.tensor(A, dtype=torch.float32)
B_torch = torch.tensor(B, dtype=torch.float32)

# Matrix multiplication
AB_torch = A_torch @ B_torch

# Print result
print("PyTorch A @ B:\n", AB_torch.numpy())
