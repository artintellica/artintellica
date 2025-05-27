"""
exercise_1_taylor_centering.py
-------------------------------------------------
Compare 4th‑degree Taylor polynomials of e^x centered at:

  • a = 0  (Maclaurin)         T4_0(x)
  • a = 1                      T4_1(x)

Plot absolute error on the interval [−3, 3] and print the
max error for each − which one has the wider “good” radius?
"""

import numpy as np
import matplotlib.pyplot as plt
from math import factorial, e

# ------------------------------------------------
# 1. helper: degree‑4 Taylor poly of e^x at center a
# ------------------------------------------------
def taylor_e_deg4(x, a=0.0):
    """
    Return T_4^(a)(x) for e^x:
      T_4(x) = Σ_{k=0}^4 e^a (x‑a)^k / k!
    """
    dx = x - a
    coeff = e ** a
    total = 0.0
    for k in range(5):               # 0..4
        total += coeff * dx**k / factorial(k)
    return total

# ------------------------------------------------
# 2. domain and ground truth
# ------------------------------------------------
xs   = np.linspace(-3, 3, 601)
true = np.exp(xs)

# T4 at a = 0 and a = 1
t4_0 = taylor_e_deg4(xs, a=0.0)
t4_1 = taylor_e_deg4(xs, a=1.0)

err_0 = np.abs(true - t4_0)
err_1 = np.abs(true - t4_1)

# ------------------------------------------------
# 3. plot
# ------------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(xs, err_0, label=r"$|e^x - T_4^{(a=0)}(x)|$")
plt.plot(xs, err_1, label=r"$|e^x - T_4^{(a=1)}(x)|$")
plt.yscale("log")
plt.xlabel("x")
plt.ylabel("absolute error (log scale)")
plt.title("Degree‑4 Taylor error for e^x: a=0 vs a=1")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 4. numeric summary
# ------------------------------------------------
print("max error on [-3,3]")
print(f"  a = 0  : {err_0.max():.3e}")
print(f"  a = 1  : {err_1.max():.3e}")

# optional: interval where error < 1e‑2
mask_0 = err_0 < 1e-2
mask_1 = err_1 < 1e-2
coverage_0 = np.ptp(xs[mask_0]) if np.any(mask_0) else 0
coverage_1 = np.ptp(xs[mask_1]) if np.any(mask_1) else 0
print("\nwidth of interval where error < 1e‑2:")
print(f"  a = 0  : {coverage_0:.2f}")
print(f"  a = 1  : {coverage_1:.2f}")
