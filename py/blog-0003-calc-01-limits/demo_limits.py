# demo_limits.py
import numpy as np
import matplotlib.pyplot as plt

# 1. sample points around 0
x = np.linspace(-1e-1, 1e-1, 2001)
y = np.where(x != 0, np.sin(x)/x, 1.0)  # define f(0)=1 by continuity

# 2. plot
plt.figure(figsize=(5,4))
plt.plot(x, y, label=r'$\,\sin x / x\,$')
plt.scatter([0], [1], color='black', zorder=3)  # the limiting point
plt.axhline(1, linestyle='--', linewidth=0.7)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Zoom on  sin(x)/x  near 0")
plt.tight_layout()
plt.show()
