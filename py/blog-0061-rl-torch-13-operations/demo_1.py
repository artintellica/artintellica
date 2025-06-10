import torch

# Define two vectors
u: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
v: torch.Tensor = torch.tensor([4.0, 5.0, 6.0])

# Element-wise addition
w: torch.Tensor = u + v
print("u:", u)
print("v:", v)
print("u + v:", w)
