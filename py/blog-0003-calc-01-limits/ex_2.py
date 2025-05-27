import numpy as np
import matplotlib.pyplot as plt

# 1. sample points around 0
x = np.linspace(-1e-1, 1e-1, 2001)
y = np.abs(x)

# 2. plot
plt.figure(figsize=(5, 4))
plt.plot(x, y, label=r"$\,\abs x \,$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Zoom on  abs(x)  near 0")
plt.tight_layout()
plt.show()
