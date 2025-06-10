import torch

M: torch.Tensor = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)  # Shape: (4, 2)
v: torch.Tensor = torch.tensor([2, 4, 6, 8]).unsqueeze(1) # Shape: (4, 1)
M_plus_v: torch.Tensor = M + v  # Broadcasting: Add v to each column of M
print("Matrix M:\n", M)
print("Vector v:\n", v)
print("Result of M + v:\n", M_plus_v)
