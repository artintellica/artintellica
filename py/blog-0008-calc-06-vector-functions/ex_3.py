#!/usr/bin/env python3
"""
exercise_3_gd_path.py
-------------------------------------------------
Gradient‑descent trajectory for

    f(x,y) = x² + y²

• start point  : (-1.8,  1.6)
• step size η  : 0.1
• stop when ‖∇f‖ < 1e‑3  or  100 iterations.

The script:
1. draws the contour + quiver plot of ∇f (as in Demo ①),
2. overlays the GD path with arrows and points.
"""

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------
# 1. analytic gradient
# ------------------------------------------------
def grad_f(x, y):
    return np.array([2 * x, 2 * y])  # [∂f/∂x, ∂f/∂y]


# ------------------------------------------------
# 2. build gradient‑descent trajectory
# ------------------------------------------------
eta = 0.1
max_iter = 100
tol_norm = 1e-3
start = np.array([-1.8, 1.6])

path = [start]
x, y = start

for _ in range(max_iter):
    g = grad_f(x, y)
    if np.linalg.norm(g) < tol_norm:
        break
    x, y = np.array([x, y]) - eta * g
    path.append([x, y])

path = np.array(path)  # shape [T, 2]

print(f"Converged in {len(path)-1} steps; final point = {path[-1]}")

# ------------------------------------------------
# 3. background: contour + quiver of ∇f
# ------------------------------------------------
xv = np.linspace(-2, 2, 21)
yv = np.linspace(-2, 2, 21)
X, Y = np.meshgrid(xv, yv)
U = 2 * X
V = 2 * Y
F = X**2 + Y**2

plt.figure(figsize=(6, 6))
plt.contour(X, Y, F, levels=15, cmap="gray", linewidths=0.6)
plt.quiver(X, Y, U, V, color="tab:blue", alpha=0.7, scale=40)

# ------------------------------------------------
# 4. overlay gradient‑descent path
# ------------------------------------------------
plt.plot(path[:, 0], path[:, 1], "-o", color="red", label="GD path (η=0.1)")
plt.scatter(
    path[0, 0],
    path[0, 1],
    color="red",
    edgecolor="black",
    zorder=5,
    s=80,
    label="start",
)
plt.scatter(0, 0, color="green", zorder=5, s=60, label="minimum")

plt.title(r"Gradient‑descent path on  $f=x^{2}+y^{2}$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()
