import torch

print("PyTorch version:", torch.__version__)

a: torch.Tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
print("a:", a)  # Tensor data

if torch.cuda.is_available():
    device: torch.device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
