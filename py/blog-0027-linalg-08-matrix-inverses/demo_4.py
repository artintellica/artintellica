import torch

# Convert to PyTorch tensors
A_torch = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
b_torch = torch.tensor([5.0, 4.0])

# Solve using torch.linalg.solve
x_torch = torch.linalg.solve(A_torch, b_torch)

# Print result
print("PyTorch solution x:", x_torch.numpy())
