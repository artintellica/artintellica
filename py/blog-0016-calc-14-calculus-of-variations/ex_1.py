"""
exercise_1_vary_functional_k.py
-------------------------------------------------
Try f(x) = sin(k pi x) for different k.
How does the regularizer J = ∫ (f')^2 dx change with frequency?
"""

import numpy as np
import matplotlib.pyplot as plt

# Discretize domain
N = 400
x = np.linspace(0, 1, N)
dx = x[1] - x[0]

ks = [1, 2, 3, 5, 8, 13]
Js = []

plt.figure(figsize=(9, 6))
for k in ks:
    f = np.sin(k * np.pi * x)
    # First derivative (analytical for comparison)
    dfdx_true = k * np.pi * np.cos(k * np.pi * x)
    # Numerical derivative (central difference)
    dfdx = np.zeros_like(f)
    dfdx[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
    dfdx[0] = (f[1] - f[0]) / dx
    dfdx[-1] = (f[-1] - f[-2]) / dx
    # Regularizer J = ∫ (f')^2 dx
    J = np.sum(dfdx**2) * dx
    Js.append(J)
    plt.plot(x, f, label=f"$k$={k}, $J$={J:.2f}")

plt.title(r"$f(x) = \sin(k\pi x)$ for different $k$; Smoother $\to$ lower $J$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(ks, Js, "o-", label=r"$J = \int (f')^2 dx$")
plt.xlabel("k (frequency)")
plt.ylabel(r"$J$ (smoothness penalty)")
plt.title("Regularizer $J$ vs frequency $k$")
plt.grid()
plt.tight_layout()
plt.show()

print("k values: ", ks)
print("J values: ", [f"{j:.3f}" for j in Js])
