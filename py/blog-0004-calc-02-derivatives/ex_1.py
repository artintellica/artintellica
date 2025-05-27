"""
Accuracy experiment: centered finite–difference derivative of sin(x).

For h = 10⁻² … 10⁻⁸ we measure the maximum absolute error
against the true derivative cos(x) on 1 001 points in [−π, π].
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# 1. reference data
xs = np.linspace(-math.pi, math.pi, 1001)
true_grad = np.cos(xs)

# 2. sweep over step sizes
hs, max_errs = [], []
for k in range(2, 9):  # 10⁻² … 10⁻⁸
    h = 10.0**-k
    fd = (np.sin(xs + h) - np.sin(xs - h)) / (2 * h)
    err = np.max(np.abs(fd - true_grad))
    hs.append(h)
    max_errs.append(err)
    print(f"h = 1e-{k:<2d}  →  max |error| = {err:.3e}")

# 3. plot error curve (log–log)
plt.figure(figsize=(6, 4))
plt.loglog(hs, max_errs, marker="o")
plt.xlabel("step size  h")
plt.ylabel("max abs error")
plt.title("Finite‑difference derivative accuracy for sin(x)")
plt.tight_layout()
plt.show()
