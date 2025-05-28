#!/usr/bin/env python3
"""
exercise_1_ellipse_work.py
-------------------------------------------------
Compute the work done by F(x, y) = [-y, x]
along the ellipse  x=2*cos(t), y=sin(t),  t ∈ [0, 2π].
Compare to analytic result.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Parametric path: ellipse
N = 600
t = np.linspace(0, 2 * np.pi, N)
x = 2 * np.cos(t)
y = np.sin(t)

# --- Field at (x, y): F = [-y, x]
Fx = -y
Fy = x

# --- Velocity vector dr/dt
dxdt = -2 * np.sin(t)
dydt = np.cos(t)

# --- Dot product F · dr/dt
dots = Fx * dxdt + Fy * dydt

# --- Integrate over t
work = np.trapz(dots, t)
print(f"Work along ellipse: {work:.5f}")

# --- Analytic result
# For ellipse x=a*cos(t), y=b*sin(t), work is 2π*a*b.
analytic = 2 * np.pi * 2 * 1
print(f"Analytic result:    {analytic:.5f}")

# --- Plot ellipse and field streamlines
xv, yv = np.meshgrid(np.linspace(-2.2, 2.2, 28), np.linspace(-1.2, 1.2, 28))
U = -yv
V = xv
plt.figure(figsize=(7, 4.5))
plt.streamplot(xv, yv, U, V, color="gray", density=1.1, linewidth=0.7, arrowsize=1)
plt.plot(x, y, "r", label="Ellipse path")
plt.scatter([0], [0], color="k", s=35, label="Origin")
plt.title(r"Work along ellipse in $\mathbf{F}(x,y)=[-y,x]$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
