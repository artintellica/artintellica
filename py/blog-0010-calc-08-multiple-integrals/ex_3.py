#!/usr/bin/env python3
"""
exercise_3_mc_non_gauss_integrand.py
-------------------------------------------------
Estimate

    I = ∬_R tanh(x+y) · p(x, y) dx dy

where p(x, y) is standard 2D Gaussian, R = [−1, 1] × [−1, 1].

Compare Monte Carlo estimate to scipy.integrate.dblquad result.
"""

import numpy as np
from scipy.integrate import dblquad

mux, muy = 0.0, 0.0
sigx, sigy = 1.0, 1.0


def pxy(x, y):
    return (
        1.0
        / (2 * np.pi * sigx * sigy)
        * np.exp(-0.5 * ((x - mux) ** 2 / sigx**2 + (y - muy) ** 2 / sigy**2))
    )


# Monte Carlo integration
N = 200_000
a, b = -1, 1
c, d = -1, 1
area = (b - a) * (d - c)

xs = np.random.uniform(a, b, N)
ys = np.random.uniform(c, d, N)
vals = np.tanh(xs + ys) * pxy(xs, ys)
mc_est = area * np.mean(vals)


# Closed-form via numerical integration (dblquad)
def integrand(y, x):
    return np.tanh(x + y) * pxy(x, y)


analytic, abserr = dblquad(integrand, a, b, lambda x: c, lambda x: d, epsabs=1e-8)

print(f"Monte Carlo estimate : {mc_est:.6f}")
print(f"Numerical integral   : {analytic:.6f}  (dblquad, abs err ≈ {abserr:.2e})")
print(f"Abs. error           : {abs(mc_est - analytic):.2e}")
