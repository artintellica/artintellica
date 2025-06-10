import torch

# Create a 2x3 matrix (2 rows, 3 columns)
A: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("Matrix A:\n", A)
print("Shape of A:", A.shape)

# Create two 2x3 matrices for element-wise operations
C: torch.Tensor = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])

# Element-wise addition
D_add: torch.Tensor = A + C
print("Element-wise addition (A + C):\n", D_add)

# Element-wise multiplication (Hadamard product)
D_mul: torch.Tensor = A * C
print("Element-wise multiplication (A * C):\n", D_mul)
