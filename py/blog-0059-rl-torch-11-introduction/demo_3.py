import torch

# Create a 1D tensor of floats, from 0 to 4
x: torch.Tensor = torch.arange(5, dtype=torch.float32)

# Detect GPUs or MPS device (Apple Silicon)
if torch.cuda.is_available():
    device: torch.device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

x_gpu: torch.Tensor = x.to(device)
print("x is on device:", x_gpu.device)

y: torch.Tensor = torch.ones(5, dtype=torch.float32, device=device)
z: torch.Tensor = x_gpu + y  # elementwise add
print("z:", z)
