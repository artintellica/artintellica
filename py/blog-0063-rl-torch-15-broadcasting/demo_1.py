import torch

A: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
row: torch.Tensor = torch.tensor([10.0, 20.0, 30.0])  # Shape: (3,)

# Broadcasting: Add row to each row of A
A_plus_row: torch.Tensor = A + row
print("Matrix A:\n", A)
print("Row vector:", row)
print("Result of A + row:\n", A_plus_row)
