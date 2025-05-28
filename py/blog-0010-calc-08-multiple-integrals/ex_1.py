#!/usr/bin/env python3
"""
exercise_1_mc_gauss2d_contour.py
-------------------------------------------------
Visualize the 2D Gaussian

    p(x, y) = N((x, y); (0,0), I)

over (x, y) ∈ [−3, 3]², with the integration box [−1, 1] × [−1, 1] overlaid.
"""

import numpy as np
import matplotlib.pyplot as plt

# Gaussian parameters
mux, muy = 0.0, 0.0
sigx, sigy = 1.0, 1.0


def pxy(x, y):
    return (
        1.0
        / (2 * np.pi * sigx * sigy)
        * np.exp(-0.5 * ((x - mux) ** 2 / sigx**2 + (y - muy) ** 2 / sigy**2))
    )


# Grid for contour plot
xv = np.linspace(-3, 3, 200)
yv = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(xv, yv)
Z = pxy(X, Y)

# Plot
plt.figure(figsize=(6, 6))
cont = plt.contourf(X, Y, Z, levels=30, cmap="viridis")
plt.colorbar(label="$p(x, y)$")

# Mark integration rectangle
a, b = -1, 1
c, d = -1, 1
rect = plt.Rectangle(
    (a, c),
    b - a,
    d - c,
    edgecolor="red",
    facecolor="none",
    linewidth=2,
    label="Integration box",
)
plt.gca().add_patch(rect)

plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Gaussian $p(x, y)$ with Integration Box $[-1, 1]^2$")
plt.legend()
plt.tight_layout()
plt.show()
