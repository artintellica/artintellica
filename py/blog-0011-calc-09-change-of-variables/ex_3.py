"""
exercise_3_volume_transformation.py
-------------------------------------------------
Sample 10,000 points z ∈ [−2,2]², transform with
    x = f(z) = tanh(Az + b),
and plot before and after in (z1, z2) and (x1, x2) planes.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# --- Parameters for nonlinear flow
A = torch.tensor([[1.2, 0.7], [-0.6, 1.5]], dtype=torch.float64)
b = torch.tensor([0.5, -0.8], dtype=torch.float64)


def f(z):
    # z: [N, 2] torch tensor
    return torch.tanh(z @ A.T + b)


# --- Sample points in square region
N = 10_000
z_np = np.random.uniform(-2, 2, size=(N, 2))
z = torch.tensor(z_np, dtype=torch.float64)
x = f(z).detach().numpy()

# --- Plot before and after
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(z_np[:, 0], z_np[:, 1], s=4, alpha=0.5)
plt.title("z-space (before flow)")
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.xlim(-2.1, 2.1)
plt.ylim(-2.1, 2.1)

plt.subplot(1, 2, 2)
plt.scatter(x[:, 0], x[:, 1], s=4, alpha=0.5, color="orange")
plt.title("x-space (after flow)")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xlim(-1.05, 1.05)
plt.ylim(-1.05, 1.05)

plt.tight_layout()
plt.show()
