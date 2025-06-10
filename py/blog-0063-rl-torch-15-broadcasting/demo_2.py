import torch

A: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
col: torch.Tensor = torch.tensor([[2.0], [3.0], [4.0]])  # Shape: (3, 1)

# Broadcasting: Multiply each row of A by the corresponding element in col
A_times_col: torch.Tensor = A * col
print("Matrix A:\n", A)
print("Column vector:\n", col)
print("Result of A * col:\n", A_times_col)
