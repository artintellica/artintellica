import torch

t: torch.Tensor = torch.tensor(42, dtype=torch.float32)
print("t:", t)  # Tensor data
t_py: float = t.item()
print("t as Python float:", t_py)  # Convert to Python float

u: torch.Tensor = torch.tensor([3, 1, 4, 1, 5, 9], dtype=torch.float32)
print("u:", u)  # Tensor data

u_col: torch.Tensor = u.unsqueeze(1)  # Reshape to a column vector (n, 1)
print("u reshaped to 2D (shape):", u_col.shape)
print("u reshaped to 2D (data):", u_col)  # Reshaped tensor data
u_back: torch.Tensor = u_col.squeeze()  # Remove the added dimension
print("u reshaped back to 1D (shape):", u_back.shape)
print("u reshaped back to 1D (data):", u_back)  # Reshaped back to 1D tensor data
