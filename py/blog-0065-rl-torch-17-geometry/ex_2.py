import torch
import matplotlib.pyplot as plt

# - Create tensors `a = [1, 7, 2, 5]` and `b = [5, 1, 2, -1]` (float).
# - Compute the Euclidean distance between them.
a: torch.Tensor = torch.tensor([1.0, 7.0, 2.0, 5.0])
b: torch.Tensor = torch.tensor([5.0, 1.0, 2.0, -1.0])
dist: torch.Tensor = torch.dist(a, b, p=2)
print("a:", a)
print("b:", b)
print("Euclidean distance between a and b:", dist.item())
