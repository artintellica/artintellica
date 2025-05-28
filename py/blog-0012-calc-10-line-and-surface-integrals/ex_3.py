"""
exercise_3_custom_streamplot.py
-------------------------------------------------
Visualize the streamplot for
    F(x, y) = [sin(y), cos(x)]
over [-2, 2]^2.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Grid over [-2, 2]^2
xv = np.linspace(-2, 2, 36)
yv = np.linspace(-2, 2, 36)
X, Y = np.meshgrid(xv, yv)

# --- Vector field
U = np.sin(Y)
V = np.cos(X)

plt.figure(figsize=(6, 6))
plt.streamplot(
    X, Y, U, V, color=np.hypot(U, V), cmap="plasma", density=1.2, linewidth=1
)
plt.title(r"Streamplot for $\mathbf{F}(x, y) = [\sin y, \cos x]$")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.colorbar(label="speed ($|\mathbf{F}|$)")
plt.tight_layout()
plt.show()
