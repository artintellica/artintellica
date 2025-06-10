import torch

# Create a 2x3 matrix (2 rows, 3 columns)
A: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("Matrix A:\n", A)
print("Shape of A:", A.shape)

# 1D vector to broadcast
vec: torch.Tensor = torch.tensor([10.0, 20.0, 30.0])
print("Vector to broadcast:", vec)
print("Shape of vec:", vec.shape)

# Broadcasting: Add vec to each row of A
result_broadcast: torch.Tensor = A + vec
print("A + broadcasted vec:\n", result_broadcast)
