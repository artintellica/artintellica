import torch

a: torch.Tensor = torch.arange(0, 12, dtype=torch.float32)
print("a:", a)
b: torch.Tensor = a.reshape(3, 4)
print("b:", b)

c: torch.Tensor = torch.rand(2 * 5, dtype=torch.float32)
print("c:", c)
d: torch.Tensor = c.reshape(2, 5)
print("d:", d)
