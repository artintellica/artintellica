import torch

a: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
b: torch.Tensor = torch.tensor([4.0, 6.0, 8.0])
dist: torch.Tensor = torch.dist(a, b, p=2)
print("a:", a)
print("b:", b)
print("Euclidean distance between a and b:", dist.item())
# or equivalently...
alt_dist: torch.Tensor = torch.norm(a - b)
print("Alternative distance (torch.norm):", alt_dist.item())
