import torch

t: torch.Tensor = torch.tensor(42, dtype=torch.float32)
print("t:", t)  # Tensor data
t_py: float = t.item()
print("t as Python float:", t_py)  # Convert to Python float

u: torch.Tensor = torch.tensor([3, 1, 4, 1, 5, 9], dtype=torch.float32)
print("u:", u)  # Tensor data
