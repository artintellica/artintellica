import torch

v: torch.Tensor = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
print("v:", v)  # Tensor data
print("v first element:", v[0])  # Access first element
print("v last element:", v[-1])  # Access last element

# Slice out every other element (e.g., elements at even indices).
u: torch.Tensor = v[::2]
print("u (every other element):", u)
print("u shape:", u.shape)  # Shape of the sliced tensor

# Reverse the tensor
reversed_v: torch.Tensor = v.flip(0)
print("Reversed v:", reversed_v)
