"""
exercise_2_tanh_maclaurin.py
-------------------------------------------------
Degree‑5 Maclaurin polynomial for tanh x:

    tanh(x) ≈  x − x³/3 + 2x⁵/15          (error O(x⁷))

Plots the true tanh and the polynomial on [−2,2],
and marks the maximum absolute error.
"""

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------
# 1. degree‑5 Maclaurin for tanh x
#    (derived from the power‑series of sinh/cosh)
# ------------------------------------------------
def tanh_deg5(x):
    return x - x**3 / 3 + (2 * x**5) / 15


# ------------------------------------------------
# 2. domain and data
# ------------------------------------------------
xs = np.linspace(-2, 2, 501)
true = np.tanh(xs)
approx = tanh_deg5(xs)
err = np.abs(true - approx)
max_err = err.max()
x_max = xs[np.argmax(err)]

# ------------------------------------------------
# 3. plot function & approximation
# ------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(xs, true, label="tanh(x)", linewidth=2)
plt.plot(xs, approx, label="degree‑5 Maclaurin", linestyle="--")
plt.scatter(
    [x_max],
    [approx[np.argmax(err)]],
    color="red",
    zorder=3,
    label=f"max error ≈ {max_err:.3e}",
)
plt.xlabel("x")
plt.ylabel("value")
plt.title("tanh x vs. degree‑5 Maclaurin approximation")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Maximum |tanh(x) - T5(x)| on [-2,2] is {max_err:.3e} at x ≈ {x_max:.2f}")
