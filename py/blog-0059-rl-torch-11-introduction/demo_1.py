import torch

# Create a 1D tensor of floats, from 0 to 4
x: torch.Tensor = torch.arange(5, dtype=torch.float32)
print("x:", x)  # Tensor data
print("Shape:", x.shape)
print("Dtype:", x.dtype)
