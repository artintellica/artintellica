import torch

M5: torch.Tensor = torch.rand(4 * 3, dtype=torch.float32).reshape(4, 3)
v1: torch.Tensor = torch.rand(3, dtype=torch.float32)

add_M5_plus_v1: torch.Tensor = M5 + v1

print("M5:\n", M5)
print("v1:\n", v1)
print("M5 + v1:\n", add_M5_plus_v1)
