# calc-05-taylor/taylor_error_exp.py
import numpy as np
import matplotlib.pyplot as plt
from math import factorial


def maclaurin_exp(x, n):
    """Return T_n(x) for e^x."""
    return sum((x**k) / factorial(k) for k in range(n + 1))


xs = np.linspace(-3, 3, 400)
true = np.exp(xs)

plt.figure(figsize=(6, 4))
for n in [1, 2, 4, 6, 8]:
    approx = maclaurin_exp(xs, n)
    err = np.abs(approx - true)
    plt.plot(xs, err, label=f"n={n}")

plt.yscale("log")
plt.xlabel("x")
plt.ylabel("|e^x - T_n(x)| (log)")
plt.title("Absolute error of Maclaurin truncations of e^x")
plt.legend()
plt.tight_layout()
plt.show()
