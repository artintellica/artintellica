import torch

M: torch.Tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)  # Shape: (3, 4)
v: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])  # Shape: (4,)
M_plus_v: torch.Tensor = M + v  # Broadcasting: Add v to each row of M
print("Matrix M:\n", M)
print("Vector v:", v)
print("Result of M + v:\n", M_plus_v)
