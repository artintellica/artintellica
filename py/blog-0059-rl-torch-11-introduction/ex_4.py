import torch

print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    device: torch.device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

x: torch.Tensor = torch.ones(6, dtype=torch.float32, device=device)
y: torch.Tensor = torch.zeros(6, dtype=torch.float32, device=device)
z: torch.Tensor = x + y  # elementwise add
print("z:", z)
