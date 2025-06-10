import torch

# A: 2x3 matrix
A: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# B: 3x2 matrix
B: torch.Tensor = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

# Method 1: Using "@" operator
C1: torch.Tensor = A @ B
print("A @ B:\n", C1)

# Method 2: Using torch.matmul
C2: torch.Tensor = torch.matmul(A, B)
print("torch.matmul(A, B):\n", C2)
