import torch

print("PyTorch version:", torch.__version__)

a: torch.Tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
print("a:", a)  # Tensor data
