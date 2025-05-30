import numpy as np
import torch

# Matrix and vector from Exercise 2
A_np = np.array([[2, 1, -1], [1, 3, 2], [0, 1, 4]])
b_np = np.array([8, 7, 3])

# Convert to PyTorch tensors
A_torch = torch.tensor(A_np, dtype=torch.float32)
b_torch = torch.tensor(b_np, dtype=torch.float32)

# Solve using PyTorch
x_torch = torch.linalg.solve(A_torch, b_torch)

# Solve using NumPy for verification
x_np = np.linalg.solve(A_np, b_np)

# Verify solutions match
solutions_match = np.allclose(x_torch.numpy(), x_np)

# Print results
print("Matrix A (NumPy):\n", A_np)
print("\nVector b (NumPy):", b_np)
print("\nSolution x (PyTorch):", x_torch.numpy())
print("\nSolution x (NumPy):", x_np)
print("\nSolutions match?", solutions_match)
