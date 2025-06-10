import torch

# Create a 2x3 matrix (2 rows, 3 columns)
A: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("Matrix A:\n", A)
print("Shape of A:", A.shape)

# Transpose matrix A (2x3 -> 3x2)
A_transpose: torch.Tensor = A.T
print("Transpose of A (A^T):\n", A_transpose)
print("Shape of A^T:", A_transpose.shape)
