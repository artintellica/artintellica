#!/usr/bin/env python3
"""
exercise_2_hessian_quadratic.py
-------------------------------------------------
For  f(x) = x^T A x  (with x ∈ ℝ², A symmetric 2x2),
compute the Hessian analytically and check it against
autograd.
Analytic result: Hessian = A + A^T = 2A (since A symmetric)
"""

import torch

torch.set_default_dtype(torch.float64)

# ---- Random symmetric 2x2 matrix ----
A = torch.randn(2, 2)
A = (A + A.t()) / 2  # symmetrize


# ---- Function f(x) = x^T A x ----
def f(x):
    return x @ A @ x


# ---- Point to test ----
x0 = torch.randn(2, requires_grad=True)

# ---- Analytic Hessian ----
H_analytic = A + A.t()  # = 2A for symmetric A


# ---- Autograd Hessian ----
def get_hessian(func, x):
    n = x.numel()
    hess = torch.zeros(n, n, dtype=x.dtype)
    grad = torch.autograd.grad(func(x), x, create_graph=True)[0]
    for i in range(n):
        grad2 = torch.autograd.grad(grad[i], x, retain_graph=True)[0]
        hess[i] = grad2
    return hess


H_auto = get_hessian(f, x0).detach()

print("A =\n", A.numpy())
print("\nAnalytic Hessian =\n", H_analytic.numpy())
print("\nAutograd Hessian =\n", H_auto.numpy())
print("\nDifference =\n", (H_analytic - H_auto).numpy())

# Confirm allclose
assert torch.allclose(H_analytic, H_auto, atol=1e-8), "Hessian mismatch!"
print("\n✓  Hessian matches analytic result for quadratic form")
