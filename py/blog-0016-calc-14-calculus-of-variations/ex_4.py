#!/usr/bin/env python3
"""
exercise_4_physics_informed_gradient.py
-------------------------------------------------
Functional   J[f] = ∫ (f'(x) - g(x))² dx
• derive δJ/δf = -2 ( f''(x) - g'(x) )
• implement J and its functional gradient on a grid
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# grid
# ------------------------------------------------------------------
N = 400
x = np.linspace(0, 1, N)
dx = x[1] - x[0]

# some arbitrary g(x)  (can be any known target derivative)
g = np.cos(2 * np.pi * x)  # example: cos(2πx)
# its derivative  g'(x) = -2π sin(2πx)
gprime = -2 * np.pi * np.sin(2 * np.pi * x)

# initial f(x): noisy sine (or anything you like)
rng = np.random.default_rng(0)
f = 0.4 * np.sin(2 * np.pi * x) + 0.2 * rng.standard_normal(N)


# helpers ----------------------------------------------------------
def first_derivative(f, dx):
    """central diff, Neumann BC"""
    d = np.empty_like(f)
    d[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
    d[0] = (f[1] - f[0]) / dx
    d[-1] = (f[-1] - f[-2]) / dx
    return d


def second_derivative(f, dx):
    """Laplacian, Neumann BC"""
    d2 = np.empty_like(f)
    d2[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / dx**2
    d2[0] = (f[1] - f[0]) / dx**2
    d2[-1] = (f[-2] - f[-1]) / dx**2
    return d2


def J_and_grad(f, g, gprime, dx):
    dfdx = first_derivative(f, dx)
    J = np.sum((dfdx - g) ** 2) * dx
    grad = -2 * (second_derivative(f, dx) - gprime)
    return J, grad


# compute -----------------------------------------------------------
J_val, grad = J_and_grad(f, g, gprime, dx)
print(f"Functional J = {J_val:.5f}")
print(f"grad shape {grad.shape}, max |grad| = {np.max(np.abs(grad)):.3e}")

# plot --------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(x, f, label="f(x)")
plt.plot(x, np.cumsum(g) * dx, "--", label="integral of g (for intuition)")
plt.legend()
plt.title("Function f and target derivative g")

plt.subplot(2, 1, 2)
plt.plot(x, grad, "r", label=r"δJ/δf = $-2(f''-g')$")
plt.axhline(0, color="k", lw=0.7)
plt.legend()
plt.xlabel("x")
plt.ylabel("functional grad")
plt.tight_layout()
plt.show()
