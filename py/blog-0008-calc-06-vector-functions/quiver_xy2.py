import numpy as np
import matplotlib.pyplot as plt

# ---- build grid -------------------------------------------------------
xv = np.linspace(-2, 2, 21)
yv = np.linspace(-2, 2, 21)
X, Y = np.meshgrid(xv, yv)

# analytical gradient
U = 2 * X  # ∂f/∂x
V = 2 * Y  # ∂f/∂y

# ---- contour + quiver plot -------------------------------------------
F = X**2 + Y**2
plt.figure(figsize=(6, 6))
plt.contour(X, Y, F, levels=10, cmap="gray", linewidths=0.6)
plt.quiver(X, Y, U, V, color="tab:blue", alpha=0.8, scale=40)
plt.title(r"Gradient field  $\nabla f(x,y)$  for  $f=x^2+y^2$")
plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
