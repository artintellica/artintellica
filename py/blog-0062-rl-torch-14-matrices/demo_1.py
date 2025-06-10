import torch

# Create a 2x3 matrix (2 rows, 3 columns)
A: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("Matrix A:\n", A)
print("Shape of A:", A.shape)

# Create a 3x2 matrix using arange and reshape
B: torch.Tensor = torch.arange(6, dtype=torch.float32).reshape(3, 2)
print("Matrix B:\n", B)
print("Shape of B:", B.shape)
