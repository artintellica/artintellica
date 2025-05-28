#!/usr/bin/env python3
"""
exercise_2_custom_surface.py
-------------------------------------------------
Visualise the scalar field

        h(x,y) = x³ – 3 x y²     (Re{(x+iy)³})

* plots level‑set contours
* overlays the gradient field  ∇h = [3x²-3y²,  -6xy]
* marks the only critical point (0,0) which is a saddle
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. grid & function
# ------------------------------------------------
xv = np.linspace(-2, 2, 41)
yv = np.linspace(-2, 2, 41)
X, Y = np.meshgrid(xv, yv)

H = X**3 - 3 * X * Y**2  # surface
U = 3 * X**2 - 3 * Y**2  # ∂h/∂x
V = -6 * X * Y  # ∂h/∂y

# ------------------------------------------------
# 2. plot
# ------------------------------------------------
plt.figure(figsize=(6, 6))
# contours
plt.contour(X, Y, H, levels=20, cmap="RdGy", linewidths=0.7)
# quiver
plt.quiver(X, Y, U, V, color="tab:blue", alpha=0.8, scale=150)
# saddle point
plt.scatter([0], [0], color="black", zorder=3, s=50, label="saddle (0,0)")

plt.title(r"Contours & gradient field of  $h(x,y)=x^{3}-3xy^{2}$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()

print("Critical point (saddle): (0, 0)")
