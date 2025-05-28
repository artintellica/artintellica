"""
exercise_2_laplacian_gaussian.py
-------------------------------------------------
Compute and plot Laplacian Δf for
    f(x, y) = exp(-x^2 - y^2)
over a grid in [-2,2]^2.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Grid
N = 120
xv = np.linspace(-2, 2, N)
yv = np.linspace(-2, 2, N)
X, Y = np.meshgrid(xv, yv)

# --- Function and Laplacian
f = np.exp(-(X**2) - Y**2)

dx = xv[1] - xv[0]
dy = yv[1] - yv[0]

# Second derivatives
d2f_dx2 = np.gradient(np.gradient(f, dx, axis=1), dx, axis=1)
d2f_dy2 = np.gradient(np.gradient(f, dy, axis=0), dy, axis=0)
lap = d2f_dx2 + d2f_dy2

# --- Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(f, extent=[-2, 2, -2, 2], origin="lower", cmap="viridis")
plt.colorbar(label="$f(x,y)$")
plt.title(r"$f(x,y) = \exp(-x^2 - y^2)$")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 2, 2)
plt.imshow(lap, extent=[-2, 2, -2, 2], origin="lower", cmap="coolwarm")
plt.colorbar(label=r"$\Delta f$")
plt.title(r"Laplacian $\Delta f$ of Gaussian")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()

# --- Where is Laplacian most negative?
i, j = np.unravel_index(np.argmin(lap), lap.shape)
print(f"Laplacian Δf is most negative at (x, y) = ({xv[j]:.2f}, {yv[i]:.2f})")
print(f"Value of Δf there: {lap[i, j]:.4f}")
