"""
exercise_2_mc_gauss2d_vary_region.py
-------------------------------------------------
Monte Carlo estimate for the probability mass of a 2D standard Gaussian
inside squares of size L = 2, 3, 4 (i.e., sides [−L/2, L/2]).
Compare to the closed-form result.
"""

import numpy as np
from scipy.stats import norm

mux, muy = 0.0, 0.0
sigx, sigy = 1.0, 1.0


def pxy(x, y):
    return (
        1.0
        / (2 * np.pi * sigx * sigy)
        * np.exp(-0.5 * ((x - mux) ** 2 / sigx**2 + (y - muy) ** 2 / sigy**2))
    )


N = 200_000  # use a large N for low MC noise

for L in [2, 3, 4]:
    a, b = -L / 2, L / 2
    area = (b - a) ** 2

    # Closed-form
    prob_x = norm.cdf(b, mux, sigx) - norm.cdf(a, mux, sigx)
    prob_y = norm.cdf(b, muy, sigy) - norm.cdf(a, muy, sigy)
    closed = prob_x * prob_y

    # Monte Carlo
    xs = np.random.uniform(a, b, N)
    ys = np.random.uniform(a, b, N)
    vals = pxy(xs, ys)
    mc_est = area * np.mean(vals)

    print(
        f"L = {L:1} | Closed-form: {closed:.5f} | MC: {mc_est:.5f} | Abs. error: {abs(closed - mc_est):.2e}"
    )
