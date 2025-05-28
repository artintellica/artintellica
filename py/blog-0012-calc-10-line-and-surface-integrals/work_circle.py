# calc-10-line-surface/work_circle.py
import numpy as np
import matplotlib.pyplot as plt


# --- Vector field F(x, y) = [-y, x]
def F(x, y):
    return np.array([-y, x])


# --- Path: unit circle
N = 400
t = np.linspace(0, 2 * np.pi, N)
r = np.stack([np.cos(t), np.sin(t)], axis=1)
drdt = np.stack([-np.sin(t), np.cos(t)], axis=1)
Fs = np.stack([F(x, y) for x, y in r])

# Compute dot(F, dr/dt) at each t
dots = np.sum(Fs * drdt, axis=1)
work = np.trapz(dots, t)
print(f"Work along circle: {work:.5f} (analytic = {2 * np.pi:.5f})")

# --- Plot field, path, streamplot
xv, yv = np.meshgrid(np.linspace(-1.3, 1.3, 24), np.linspace(-1.3, 1.3, 24))
U, V = F(xv, yv)
plt.figure(figsize=(6, 6))
plt.streamplot(xv, yv, U, V, color="gray", density=1.1, linewidth=0.7, arrowsize=1)
plt.plot(r[:, 0], r[:, 1], "r", label="Path (unit circle)")
plt.scatter([0], [0], color="k", s=35, label="Origin")
plt.title("Vector field $[-y, x]$ and circular path")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
