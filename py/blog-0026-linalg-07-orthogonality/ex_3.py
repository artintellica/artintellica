import numpy as np
import torch

# Vectors from Exercise 2
u_np = np.array([2, 3])  # Vector to project
v_np = np.array([1, 1])  # Vector to project onto

# Convert to PyTorch tensors
u_torch = torch.tensor(u_np, dtype=torch.float32)
v_torch = torch.tensor(v_np, dtype=torch.float32)

# Compute projection using PyTorch
dot_uv_torch = torch.dot(u_torch, v_torch)  # Dot product u · v
norm_v_squared_torch = torch.sum(v_torch**2)  # Squared length of v
projection_torch = (dot_uv_torch / norm_v_squared_torch) * v_torch

# Compute projection using NumPy for verification
dot_uv_np = np.dot(u_np, v_np)
norm_v_squared_np = np.sum(v_np**2)
projection_np = (dot_uv_np / norm_v_squared_np) * v_np

# Print results
print("Vector u (NumPy):", u_np)
print("Vector v (NumPy):", v_np)
print("\nProjection (PyTorch):", projection_torch.numpy())
print("Projection (NumPy):", projection_np)
print(
    "\nDo PyTorch and NumPy results match?",
    np.allclose(projection_torch.numpy(), projection_np),
)
