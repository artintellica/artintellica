import torch

# Define two vectors
u: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
v: torch.Tensor = torch.tensor([4.0, 5.0, 6.0])

# Manual dot product using element-wise multiplication and sum
manual_dot: torch.Tensor = (u * v).sum()
print("Manual dot product (u · v):", manual_dot.item())

# Built-in dot product
builtin_dot: torch.Tensor = torch.dot(u, v)
print("Built-in dot product (u · v):", builtin_dot.item())
