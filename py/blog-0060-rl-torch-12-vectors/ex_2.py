import torch

v: torch.Tensor = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
print("v:", v)  # Tensor data
print("v first element:", v[0])  # Access first element
print("v last element:", v[-1])  # Access last element
