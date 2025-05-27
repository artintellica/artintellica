# calc-05-taylor/animate_exp.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import factorial

# domain & true curve
xs = np.linspace(-3, 3, 400)
true = np.exp(xs)

fig, ax = plt.subplots(figsize=(6, 4))
(line,) = ax.plot([], [], lw=2)
ax.plot(xs, true, "k--", label="e^x")
ax.set_xlim(-3, 3)
ax.set_ylim(-1, 20)
ax.set_title("Building e^x via Maclaurin truncations")
ax.legend()


def taylor_poly(x, n):
    return sum((x**k) / factorial(k) for k in range(n + 1))


def init():
    line.set_data([], [])
    return (line,)


def update(frame):
    y = taylor_poly(xs, frame)
    line.set_data(xs, y)
    line.set_label(f"T_{frame}(x)")
    ax.legend()
    return (line,)


anim = FuncAnimation(
    fig, update, frames=range(0, 11), init_func=init, interval=800, blit=True
)
anim.save("exp_taylor.gif", writer="pillow")
