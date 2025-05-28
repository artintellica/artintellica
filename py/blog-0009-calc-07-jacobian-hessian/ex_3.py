"""
exercise_3_saddle_plot.py
-------------------------------------------------
For  f(x,y) = x³ − 3 x y²
• print Hessian at (0,0)
• colour‑grid regions by Hessian eigen‑signs
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


# ------------------------------------------------
# 1. autograd helpers
# ------------------------------------------------
def f(xy):
    x, y = xy
    return x**3 - 3 * x * y**2


def hessian(xy):
    xy = xy.clone().detach().requires_grad_(True)
    g = torch.autograd.grad(f(xy), xy, create_graph=True)[0]
    H = torch.zeros(2, 2, dtype=xy.dtype)
    for i in range(2):
        H[i] = torch.autograd.grad(g[i], xy, retain_graph=True)[0]
    return H.detach()


# Hessian at origin
H0 = hessian(torch.tensor([0.0, 0.0]))
print("Hessian at (0,0):\n", H0.numpy())
print("Eigenvalues at (0,0):", np.linalg.eigvalsh(H0.numpy()))

# ------------------------------------------------
# 2. grid & classification
# ------------------------------------------------
xv = np.linspace(-2, 2, 41)
yv = np.linspace(-2, 2, 41)
X, Y = np.meshgrid(xv, yv)
colors = np.empty(X.shape, dtype="<U6")  # string storage

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        eigs = np.linalg.eigvalsh(hessian(torch.tensor([X[i, j], Y[i, j]])).numpy())
        pos = np.all(eigs > 1e-6)
        neg = np.all(eigs < -1e-6)
        if pos:
            colors[i, j] = "blue"
        elif neg:
            colors[i, j] = "red"
        elif np.all(np.abs(eigs) < 1e-6):
            colors[i, j] = "black"
        else:
            colors[i, j] = "green"

# ------------------------------------------------
# 3. plot coloured points
# ------------------------------------------------
plt.figure(figsize=(6, 6))
for col, lbl in [
    ("blue", "convex"),
    ("red", "concave"),
    ("green", "saddle"),
    ("black", "degenerate"),
]:
    m = colors == col
    if m.any():
        plt.scatter(X[m], Y[m], c=col, s=30, edgecolors="k", label=lbl, alpha=0.8)

plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Hessian‑sign map for  $f(x,y)=x^3-3xy^2$")
plt.legend()
plt.tight_layout()
plt.show()
