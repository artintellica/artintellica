"""
exercise_4_mc_convergence.py
-------------------------------------------------
Estimate the probability mass of a standard 2D Gaussian in
[−1, 1]² for N = 100 ... 1_000_000. Plot the MC estimate and
absolute error vs N (log‑x).
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Parameters for the Gaussian and box
mux, muy = 0.0, 0.0
sigx, sigy = 1.0, 1.0
a, b = -1, 1
c, d = -1, 1
area = (b - a) * (d - c)


def pxy(x, y):
    return (
        1.0
        / (2 * np.pi * sigx * sigy)
        * np.exp(-0.5 * ((x - mux) ** 2 / sigx**2 + (y - muy) ** 2 / sigy**2))
    )


# Closed form result
prob_x = norm.cdf(b, mux, sigx) - norm.cdf(a, mux, sigx)
prob_y = norm.cdf(d, muy, sigy) - norm.cdf(c, muy, sigy)
closed = prob_x * prob_y

# Range of N
Ns = np.logspace(2, 6, num=20, dtype=int)
mc_ests = []
abs_errs = []

for N in Ns:
    xs = np.random.uniform(a, b, N)
    ys = np.random.uniform(c, d, N)
    vals = pxy(xs, ys)
    mc = area * np.mean(vals)
    mc_ests.append(mc)
    abs_errs.append(abs(mc - closed))

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.plot(Ns, mc_ests, marker="o", label="MC estimate")
plt.axhline(closed, color="gray", linestyle="--", label="Closed form")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Estimate")
plt.title("Monte Carlo estimate vs N")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(Ns, abs_errs, marker="o", color="r")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("N")
plt.ylabel("Absolute error")
plt.title("Absolute error vs N (log-log)")

plt.tight_layout()
plt.show()
