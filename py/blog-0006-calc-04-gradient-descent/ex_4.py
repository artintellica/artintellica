"""
line_search_vs_fixed.py
-------------------------------------------------
Gradient descent with **exact line search** in 1‑D.

f(x) = x^4 – 3 x^2 + 2  (global minima at ±√1.5)

At each iteration we pick the step length η≥0 that
minimises  φ(η) = f(x − η f'(x))  along the search ray.
A coarse‑then‑fine grid search is plenty for this toy
example.  We compare the number of iterations needed
to reach |f − f_min| < 1e‑3 against a fixed‑η run.
"""

import numpy as np


# ------------------------------------------------
# Loss and analytic gradient
# ------------------------------------------------
def f(x: float) -> float:
    return x**4 - 3 * x**2 + 2


def grad_f(x: float) -> float:
    return 4 * x**3 - 6 * x


F_MIN = -0.25  # true global minimum
TOL = 1e-3
MAX_STEPS = 100


# ------------------------------------------------
# Coarse‑to‑fine line search  η* = argmin f(x-ηg)
# ------------------------------------------------
def line_search_eta(x: float, g: float, eta_max: float = 2.0) -> float:
    """Return η that (approximately) minimises φ(η)."""
    coarse = np.linspace(0.0, eta_max, 50)
    vals = [f(x - e * g) for e in coarse]
    i_best = int(np.argmin(vals))

    # refine within neighbour interval
    left = coarse[max(i_best - 1, 0)]
    right = coarse[min(i_best + 1, len(coarse) - 1)]
    fine = np.linspace(left, right, 20)
    vals2 = [f(x - e * g) for e in fine]
    return float(fine[int(np.argmin(vals2))])


# ------------------------------------------------
# 1. Fixed‑η gradient descent
# ------------------------------------------------
def gd_fixed(x0: float, eta: float) -> int:
    x = x0
    for step in range(1, MAX_STEPS + 1):
        if abs(f(x) - F_MIN) < TOL:
            return step - 1
        x -= eta * grad_f(x)
    return MAX_STEPS


# ------------------------------------------------
# 2. Line‑search gradient descent
# ------------------------------------------------
def gd_line_search(x0: float) -> int:
    x = x0
    for step in range(1, MAX_STEPS + 1):
        if abs(f(x) - F_MIN) < TOL:
            return step - 1
        g = grad_f(x)
        eta = line_search_eta(x, g, eta_max=2.0)
        x -= eta * g
    return MAX_STEPS


# ------------------------------------------------
# Run the comparison
# ------------------------------------------------
if __name__ == "__main__":
    x_start = -2.0  # try other starts if you like
    iters_fixed = gd_fixed(x_start, eta=0.05)
    iters_ls = gd_line_search(x_start)

    print(f"Start x0 = {x_start}")
    print(f"Fixed‑η GD  (η = 0.05): {iters_fixed} iterations")
    print(f"Line‑search GD        : {iters_ls} iterations")
