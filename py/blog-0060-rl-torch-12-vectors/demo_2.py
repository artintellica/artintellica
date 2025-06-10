import torch

# Scalar (0-dimensional tensor, holds a single value)
a: torch.Tensor = torch.tensor(7.5)
print("Scalar a:", a)
print("Shape (should be torch.Size([])):", a.shape)
print("Dimensions (should be 0):", a.dim())

# 1-D vector tensor
v: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("Vector v:", v)
print("Shape:", v.shape)
print("Dimensions:", v.dim())

# Indexing: Get the second element (index 1)
second: torch.Tensor = v[1]
print("Second element:", second.item())  # .item() to extract Python float

# Slicing: Get the first three elements
first_three: torch.Tensor = v[:3]
print("First three elements:", first_three)

# Reversing the vector
reversed_v: torch.Tensor = v.flip(0)
print("Reversed v:", reversed_v)
