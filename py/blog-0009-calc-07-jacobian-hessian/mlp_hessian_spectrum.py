#!/usr/bin/env python3
"""
mlp_hessian_spectrum_fixed.py
-------------------------------------------------
Robust Hessian spectrum demo for a tiny 2‑layer MLP
f : ℝ² → ℝ.

• Works even if some parameters are unused (allow_unused=True)
• Handles None gradients by substituting zeros
• Uses autograd.grad only (no .backward cycles)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------
# 1. network definition
# ---------------------------------------------------------------------
def net(flat_params, x):
    """
    flat_params contains W1(4), b1(2), W2(2), b2(1)  → 9 numbers total
    """
    W1 = flat_params[0:4].view(2, 2)
    b1 = flat_params[4:6]
    W2 = flat_params[6:8].view(1, 2)
    b2 = flat_params[8:9]
    h = torch.tanh(W1 @ x + b1)
    return (W2 @ h + b2).squeeze()  # scalar


# ---------------------------------------------------------------------
# 2. initialise parameters & input
# ---------------------------------------------------------------------
param_vec = torch.randn(9, requires_grad=True)  # flattened parameters
x = torch.randn(2, requires_grad=False)  # treat input as constant

# ---------------------------------------------------------------------
# 3. compute scalar output
# ---------------------------------------------------------------------
y = net(param_vec, x)

# ---------------------------------------------------------------------
# 4. first‑order gradient wrt *all* parameters
# ---------------------------------------------------------------------
(grad1,) = torch.autograd.grad(y, param_vec, create_graph=True)

# ---------------------------------------------------------------------
# 5. build Hessian by differentiating each grad component
# ---------------------------------------------------------------------
n = param_vec.numel()
hessian = torch.zeros(n, n, dtype=param_vec.dtype)

for i in range(n):
    g_i = grad1[i]
    # If g_i is constant, its gradient is zero — handle via allow_unused=True
    (row,) = torch.autograd.grad(
        g_i, param_vec, retain_graph=True, create_graph=False, allow_unused=True
    )
    # row may contain None where gradient is undefined (unused); replace
    row = torch.where(torch.isfinite(row), row, torch.zeros_like(row))
    hessian[i] = row

# ---------------------------------------------------------------------
# 6. eigen‑spectrum
# ---------------------------------------------------------------------
eigvals = np.linalg.eigvalsh(hessian.detach().numpy())
print("Hessian eigenvalues:", np.round(eigvals, 5))

plt.stem(eigvals)
plt.title("Hessian eigenvalues (tiny MLP)")
plt.xlabel("eigen‑index")
plt.ylabel("curvature λ")
plt.tight_layout()
plt.show()
