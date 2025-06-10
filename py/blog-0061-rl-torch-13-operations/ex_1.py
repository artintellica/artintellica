import torch

a: torch.Tensor = torch.tensor([2, 3, 4, 5], dtype=torch.float32)
b: torch.Tensor = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
sum_ab: torch.Tensor = a + b
print("a:", a)
print("b:", b)
print("a + b:", sum_ab)
