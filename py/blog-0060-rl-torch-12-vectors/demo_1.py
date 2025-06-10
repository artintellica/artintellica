import torch

# Scalar (0-dimensional tensor, holds a single value)
a: torch.Tensor = torch.tensor(7.5)
print("Scalar a:", a)
print("Shape (should be torch.Size([])):", a.shape)
print("Dimensions (should be 0):", a.dim())

# 1-D vector tensor
v: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("Vector v:", v)
print("Shape:", v.shape)
print("Dimensions:", v.dim())

# Changing shape: Reshape vector v to a column vector (n, 1)
v_col: torch.Tensor = v.unsqueeze(1)  # adds a new dimension at position 1
print("v as column vector (shape):", v_col.shape)
print("v as column vector (data):", v_col)

# Flatten v_col back to 1D
v_flat: torch.Tensor = v_col.squeeze()
print("v_col squeezed to 1D:", v_flat.shape)
print("v_col squeezed to 1D (data):", v_flat)
