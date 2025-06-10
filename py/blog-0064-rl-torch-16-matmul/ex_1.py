import torch

# - Create matrix $M_1$ of shape $(2, 4)$ (e.g., fill with numbers from 1 to 8).
# - Create matrix $M_2$ of shape $(4, 3)$ (e.g., fill with numbers 9 to 20).
# - Multiply using both `@` and `torch.matmul`. Print both results—are they equal?

M1: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
M2: torch.Tensor = torch.tensor(
    [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0], [18.0, 19.0, 20.0]]
)
C1: torch.Tensor = M1 @ M2
C2: torch.Tensor = torch.matmul(M1, M2)
print("M1 @ M2:\n", C1)
print("torch.matmul(M1, M2):\n", C2)
print("Are the results equal?", torch.allclose(C1, C2))
