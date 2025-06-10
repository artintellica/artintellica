import torch

a: torch.Tensor = torch.arange(0, 12, dtype=torch.float32)
print("a:", a)
b: torch.Tensor = a.reshape(3, 4)
print("b:", b)
