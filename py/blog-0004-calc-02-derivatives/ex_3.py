"""
2‑D gradient field for  f(x,y) = x^2 + 0.5 y^2
∇f = (2x,  y)
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. build a grid
x_min, x_max, n = -2.5, 2.5, 21  # 21×21 grid
xs = np.linspace(x_min, x_max, n)
ys = np.linspace(x_min, x_max, n)
X, Y = np.meshgrid(xs, ys)

# 2. compute gradients
U = 2 * X  # ∂f/∂x = 2x
V = Y  # ∂f/∂y = y           (because 0.5·2 = 1)

# 3. optional contours of f for context
F = X**2 + 0.5 * Y**2

plt.figure(figsize=(6, 6))
plt.contour(X, Y, F, levels=10, cmap="gray", linewidths=0.7)
plt.quiver(X, Y, U, V, color="tab:blue", alpha=0.8, scale=20)
plt.title(r"Gradient field of  $f(x,y)=x^2 + 0.5 * y^2$")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()
