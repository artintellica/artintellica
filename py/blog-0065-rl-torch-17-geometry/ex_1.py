import torch
import matplotlib.pyplot as plt

# - Create a 1D tensor `v = [6, 8]` (float).
# - Compute and print its Euclidean norm (should be $10$).
v: torch.Tensor = torch.tensor([6.0, 8.0])
norm_v: torch.Tensor = torch.norm(v, p=2)
print("Vector v:", v)
print("Euclidean norm (||v||):", norm_v.item())
