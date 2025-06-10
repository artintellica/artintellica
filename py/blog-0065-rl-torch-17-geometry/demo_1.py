import torch

v: torch.Tensor = torch.tensor([3.0, 4.0])
norm_v: torch.Tensor = torch.norm(v, p=2)
print("Vector v:", v)
print("Euclidean norm (||v||):", norm_v.item())
