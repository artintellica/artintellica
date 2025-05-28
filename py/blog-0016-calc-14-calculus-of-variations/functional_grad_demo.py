# calc-14-functional-deriv/functional_grad_demo.py
import numpy as np
import matplotlib.pyplot as plt

# Discretize domain
N = 120
x = np.linspace(0, 1, N)
dx = x[1] - x[0]

# A sample function
f = np.sin(3 * np.pi * x) + 0.4 * np.cos(5 * np.pi * x)

# Compute first derivative (central difference)
dfdx = np.zeros_like(f)
dfdx[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
dfdx[0] = (f[1] - f[0]) / dx  # forward difference at left
dfdx[-1] = (f[-1] - f[-2]) / dx  # backward at right

# Compute second derivative (Laplacian)
d2fdx2 = np.zeros_like(f)
d2fdx2[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / dx**2
d2fdx2[0] = d2fdx2[-1] = 0.0  # Dirichlet boundary (could also use one-sided diff)

# Compute functional, and its gradient
J = np.sum(dfdx**2) * dx
func_grad = -2 * d2fdx2

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(x, f, label="f(x)")
plt.plot(x, dfdx, "--", label="f'(x)")
plt.plot(x, d2fdx2, ":", label="f''(x)")
plt.legend()
plt.title("Function, Derivatives, and Functional Gradient")
plt.subplot(2, 1, 2)
plt.plot(x, func_grad, "r", label=r"$-2f''(x)$ (functional grad)")
plt.axhline(0, color="k", lw=0.7)
plt.legend()
plt.xlabel("x")
plt.tight_layout()
plt.show()

print(f"Value of regularizer J = ∫ (f')² dx: {J:.5f}")
