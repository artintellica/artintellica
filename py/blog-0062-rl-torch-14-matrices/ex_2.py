import torch

M1: torch.Tensor = torch.arange(0, 12, dtype=torch.float32).reshape(3, 4)
M1_T: torch.Tensor = M1.T
print("M1 shape:", M1.shape)
print("M1_T shape:", M1_T.shape)
print("M1:\n", M1)
print("M1_T:\n", M1_T)

