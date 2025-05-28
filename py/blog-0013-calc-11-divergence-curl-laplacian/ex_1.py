"""
exercise_1_div_curl_grid.py
-------------------------------------------------
For F(x, y) = [y, -x], compute divergence and curl on
a grid in [-2,2]^2 and plot as images.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Grid
N = 64
xv = np.linspace(-2, 2, N)
yv = np.linspace(-2, 2, N)
X, Y = np.meshgrid(xv, yv)

# --- Vector field: F = [y, -x]
Fx = Y
Fy = -X

# --- Numerical partial derivatives
dx = xv[1] - xv[0]
dy = yv[1] - yv[0]

# ∂Fx/∂x, ∂Fy/∂y
dFx_dx = np.gradient(Fx, dx, axis=1)
dFy_dy = np.gradient(Fy, dy, axis=0)

# ∂Fy/∂x, ∂Fx/∂y
dFy_dx = np.gradient(Fy, dx, axis=1)
dFx_dy = np.gradient(Fx, dy, axis=0)

# Divergence: ∂Fx/∂x + ∂Fy/∂y
div = dFx_dx + dFy_dy

# Curl (scalar in 2D): ∂Fy/∂x - ∂Fx/∂y
curl = dFy_dx - dFx_dy

# --- Plot
plt.figure(figsize=(11, 4.5))

plt.subplot(1, 2, 1)
plt.imshow(div, extent=[-2, 2, -2, 2], origin="lower", cmap="bwr")
plt.colorbar(label="divergence")
plt.title(r"Divergence of $\mathbf{F}(x, y)=[y, -x]$")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 2, 2)
plt.imshow(curl, extent=[-2, 2, -2, 2], origin="lower", cmap="bwr")
plt.colorbar(label="curl")
plt.title(r"Curl of $\mathbf{F}(x, y)=[y, -x]$")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()

# Optional: print analytic results
print("Analytic divergence everywhere: 0")
print("Analytic curl everywhere: -2")
print("Mean of computed div:", np.mean(div))
print("Mean of computed curl:", np.mean(curl))
