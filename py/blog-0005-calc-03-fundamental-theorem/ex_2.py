"""
Verify that d/dx ∫₀ˣ tanh(t) dt  equals 1 – tanh²x
using PyTorch autograd + a differentiable trapezoid rule.
"""

import torch

torch.set_default_dtype(torch.float64)  # better accuracy


# -------------------------------
# 1. differentiable quadrature
# -------------------------------
def integral_tanh(x, n=2000):
    """
    Approximates ∫₀ˣ tanh(t) dt on a fixed grid with trapz.
    Works for a *batch* of x values (shape: [m]).
    """
    t_grid = torch.linspace(0.0, 1.0, n, device=x.device)  # [n]
    # Create [m, n] matrix: each row is t_grid * x_i
    t = torch.outer(x, t_grid)  # [m,n]
    y = torch.tanh(t)
    # trapz integrates along last dim; spacing varies with x, so pass t
    return torch.trapz(y, t, dim=-1)  # [m]


# -------------------------------
# 2. pick some test points
# -------------------------------
xs = torch.tensor([-2.0, -1.0, -0.3, 0.0, 0.5, 1.2, 2.3], requires_grad=True)  # [m]

# -------------------------------
# 3. compute integral + autograd grad
# -------------------------------
F = integral_tanh(xs)  # [m]
F.sum().backward()  # accumulate and back‑prop once
grad_auto = xs.grad  # [m]

# -------------------------------
# 4. analytic gradient
# -------------------------------
grad_true = 1.0 - torch.tanh(xs.detach()) ** 2

# -------------------------------
# 5. print comparison
# -------------------------------
print(f"{'x':>7} | {'autograd':>12} | {'analytic':>12} | abs err")
print("-" * 50)
for xi, ga, gt in zip(xs, grad_auto, grad_true):
    err = abs(ga - gt)
    print(f"{float(xi):7.2f} | {ga:12.8f} | {gt:12.8f} | {err:.2e}")
