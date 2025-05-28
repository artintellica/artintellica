# calc-08-multint/mc_gauss_2d.py
import numpy as np
from scipy.stats import norm

# --- parameters for Gaussian
mux, muy = 0.0, 0.0
sigx, sigy = 1.0, 1.0


def pxy(x, y):
    return (
        1.0
        / (2 * np.pi * sigx * sigy)
        * np.exp(-0.5 * ((x - mux) ** 2 / sigx**2 + (y - muy) ** 2 / sigy**2))
    )


# --- region: square centered at 0
a, b = -1, 1
c, d = -1, 1
area = (b - a) * (d - c)

# --- closed-form (product of 1-D CDFs)
prob_x = norm.cdf(b, mux, sigx) - norm.cdf(a, mux, sigx)
prob_y = norm.cdf(d, muy, sigy) - norm.cdf(c, muy, sigy)
closed = prob_x * prob_y

# --- Monte Carlo estimate
N = 100_000
xs = np.random.uniform(a, b, N)
ys = np.random.uniform(c, d, N)
vals = pxy(xs, ys)
mc_est = area * np.mean(vals)

print(f"Closed‑form prob in box: {closed:.5f}")
print(f"Monte Carlo estimate  : {mc_est:.5f}")
print(f"Absolute error        : {abs(closed - mc_est):.2e}")
