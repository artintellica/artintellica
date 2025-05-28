"""
exercise_2_radial_field_work.py
-------------------------------------------------
Compute the work done by F(x, y) = [x, y]
along the unit circle: x = cos(t), y = sin(t), t ∈ [0, 2π].
Compare to analytic result and explain why.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Parametric path: unit circle
N = 500
t = np.linspace(0, 2 * np.pi, N)
x = np.cos(t)
y = np.sin(t)

# --- Vector field F(x, y) = [x, y]
Fx = x
Fy = y

# --- Velocity vector dr/dt
dxdt = -np.sin(t)
dydt = np.cos(t)

# --- Dot product F · dr/dt
dots = Fx * dxdt + Fy * dydt

# --- Integrate over t
work = np.trapz(dots, t)
print(f"Work along unit circle: {work:.5f}")

# --- Analytic explanation:
# The field is always radial, and the velocity vector is always tangential,
# so their dot product is zero at every point => total work is zero.

print(
    "Analytic: The work should be 0, because the field is perpendicular to the path everywhere."
)

# --- Plot circle and field streamlines
xv, yv = np.meshgrid(np.linspace(-1.2, 1.2, 25), np.linspace(-1.2, 1.2, 25))
U = xv
V = yv
plt.figure(figsize=(6, 6))
plt.streamplot(xv, yv, U, V, color="gray", density=1.1, linewidth=0.7, arrowsize=1)
plt.plot(x, y, "r", label="Unit circle path")
plt.scatter([0], [0], color="k", s=35, label="Origin")
plt.title(r"Work along unit circle in $\mathbf{F}(x,y)=[x,y]$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
