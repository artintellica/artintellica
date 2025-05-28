# calc-10-line-surface/streamplot_grad.py
import numpy as np
import matplotlib.pyplot as plt


def gradF(x, y):
    return -2 * x, -2 * y


xv, yv = np.meshgrid(np.linspace(-2, 2, 28), np.linspace(-2, 2, 28))
U, V = gradF(xv, yv)
plt.figure(figsize=(6, 6))
plt.streamplot(xv, yv, U, V, color="blue", density=1.4, linewidth=1)
plt.title(r"Streamplot: $-\nabla f$, $f(x,y)=x^2+y^2$")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter([0], [0], color="k", s=30, label="Minimum")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
