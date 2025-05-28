#!/usr/bin/env python3
"""
exercise_4_hessian_autograd.py
-------------------------------------------------
Compute the **Hessian** of

    f(x, y) = x² + y²

using torch.autograd.grad with create_graph=True.

Check that it matches the analytic Hessian:
    [[2, 0],
     [0, 2]]
"""

import torch

torch.set_default_dtype(torch.float64)


# ------------------------------------------------
# 1. function and point
# ------------------------------------------------
def f(xy):
    x, y = xy
    return x**2 + y**2


point = torch.tensor([1.7, -0.9], requires_grad=True)

# ------------------------------------------------
# 2. compute gradient (∇f)
# ------------------------------------------------
grad = torch.autograd.grad(f(point), point, create_graph=True)[0]  # shape [2]

# ------------------------------------------------
# 3. compute Hessian (Jacobian of grad)
# ------------------------------------------------
hessian = torch.zeros(2, 2, dtype=point.dtype)
for i in range(2):
    grad_i = grad[i]
    hess_row = torch.autograd.grad(grad_i, point, retain_graph=True)[0]
    hessian[i] = hess_row

print("Autograd Hessian:\n", hessian.numpy())
print("Analytic Hessian:\n", [[2, 0], [0, 2]])
print("Difference:\n", hessian.numpy() - [[2, 0], [0, 2]])

# check allclose
assert torch.allclose(
    hessian, torch.tensor([[2.0, 0.0], [0.0, 2.0]])
), "Hessian mismatch!"
print("✓  Hessian matches analytic [[2,0],[0,2]]")
