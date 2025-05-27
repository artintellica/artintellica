"""
multi_start_hist.py
-------------------------------------------------
Run vanilla gradient descent on the 1‑D quartic

    f(x) = x^4 – 3 x^2 + 2

from 50 random starting points in [‑3, 3] for a
set of learning‑rates η.  After a fixed number of
steps, plot a histogram of the final positions and
report how many runs (if any) ended “near” the
local maximum at x = 0.

Usage
-----
$ python multi_start_hist.py
-------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# Loss and analytic gradient
# ------------------------------------------------
def f(x: float) -> float:
    return x ** 4 - 3 * x ** 2 + 2


def grad_f(x: float) -> float:
    return 4 * x ** 3 - 6 * x


# ------------------------------------------------
# One‑dimensional gradient‑descent loop
# ------------------------------------------------
def gradient_descent(x0: float, eta: float, steps: int = 60) -> float:
    """Return x_T after `steps` steps, or np.nan if divergence occurs."""
    x = x0
    for _ in range(steps):
        g = grad_f(x)
        if not np.isfinite(g):
            return np.nan
        x_next = x - eta * g
        if not np.isfinite(x_next):
            return np.nan
        x = x_next
    return x


# ------------------------------------------------
# Hyper‑parameters
# ------------------------------------------------
np.random.seed(0)
N_RUNS   = 50
X_RANGE  = (-3.0, 3.0)
ETAS     = [0.01, 0.1, 0.5]
STEPS    = 60
TOL_NEAR = 0.05            # “near 0” if |x| < TOL_NEAR

# random initial points
x0s = np.random.uniform(*X_RANGE, N_RUNS)

# ------------------------------------------------
# Run experiments
# ------------------------------------------------
results = {eta: [] for eta in ETAS}

for eta in ETAS:
    for x0 in x0s:
        xf = gradient_descent(x0, eta, steps=STEPS)
        results[eta].append(xf)

# ------------------------------------------------
# Plot histograms & print counts
# ------------------------------------------------
for eta in ETAS:
    vals = np.array(results[eta])
    finite_vals = vals[np.isfinite(vals)]

    plt.figure(figsize=(6, 4))
    plt.hist(finite_vals, bins=15, color="tab:orange", edgecolor="black")
    plt.axvline(0, linestyle="--", color="tab:gray", linewidth=1)
    plt.title(f"Final positions after GD (η = {eta})")
    plt.xlabel("x_final")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    n_near_zero = np.sum(np.abs(finite_vals) < TOL_NEAR)
    print(f"η = {eta:<4}  →  {n_near_zero} / {len(finite_vals)} "
          f"runs ended with |x| < {TOL_NEAR}")
