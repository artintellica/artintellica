import torch

# Define two vectors
u: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
v: torch.Tensor = torch.tensor([4.0, 5.0, 6.0])

# Scalar multiplication
c: float = 2.0
scaled_u: torch.Tensor = c * u
print(f"u scaled by {c}:", scaled_u)
