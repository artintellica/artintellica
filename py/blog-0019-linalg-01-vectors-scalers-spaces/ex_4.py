import numpy as np
import torch

# Define NumPy vectors
u_np = np.array([1, 2])
v_np = np.array([2, -1])

# Convert NumPy vectors to PyTorch tensors
u_torch = torch.from_numpy(u_np).float()
v_torch = torch.from_numpy(v_np).float()

# Perform vector addition using PyTorch
sum_torch = u_torch + v_torch

# Perform vector addition using NumPy for verification
sum_np = u_np + v_np

# Print results
print("NumPy vector u:", u_np)
print("NumPy vector v:", v_np)
print("PyTorch vector u:", u_torch)
print("PyTorch vector v:", v_torch)
print("PyTorch sum (u + v):", sum_torch)
print("NumPy sum (u + v):", sum_np)

# Verify the results match
match = np.allclose(sum_torch.numpy(), sum_np)
print("Do PyTorch and NumPy results match?", match)
