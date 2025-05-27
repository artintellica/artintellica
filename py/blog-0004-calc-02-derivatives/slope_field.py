# calc-02-derivatives/slope_field.py
import numpy as np
import matplotlib.pyplot as plt


def loss(x):  # toy non‑convex loss
    return 0.3 * np.sin(3 * x) + 0.5 * (x**2)


xs = np.linspace(-3, 3, 41)
ys = loss(xs)
grads = np.gradient(ys, xs)  # finite diff for visualization

plt.figure(figsize=(6, 3))
plt.plot(xs, ys, color="black")
plt.quiver(
    xs,
    ys,
    np.ones_like(grads),
    grads,
    angles="xy",
    scale_units="xy",
    scale=10,
    width=0.004,
    alpha=0.7,
)
plt.title("Loss curve with slope arrows (direction of steepest ascent)")
plt.xlabel("x")
plt.ylabel("L(x)")
plt.tight_layout()
plt.show()
