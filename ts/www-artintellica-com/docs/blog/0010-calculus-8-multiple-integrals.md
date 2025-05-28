+++
title = "Calculus 8: Multiple Integrals — Monte Carlo Meets the Multivariate Gaussian"
date  = "2025‑05‑28"
author = "Artintellica"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0010-calc-08-multiple-integrals"
+++

> _“When integrals get high‑dimensional, randomness is your friend.”_

---

## 1 · The Need for Multiple Integrals in ML

Machine learning is filled with integrals over many variables:

- **Expected loss:** $\mathbb{E}_\mathbf{x}[L(f(\mathbf{x}))]$
- **Marginal likelihood / ELBO:** $\int p(\mathbf{x},\mathbf{z})\,d\mathbf{z}$
- **Normalization constants:**
  $\int \mathcal{N}(\mathbf{x};\mu,\Sigma)\,d\mathbf{x}=1$

Most closed forms only work for the **Gaussian** and a few others. For the rest:
**numerics**!

---

## 2 · Closed‑form for the 2‑D Gaussian

The density is

$$
p(x, y) = \frac{1}{2\pi\sigma_x\sigma_y} \exp\left(-\frac{1}{2}\left[
    \left(\frac{x-\mu_x}{\sigma_x}\right)^2 +
    \left(\frac{y-\mu_y}{\sigma_y}\right)^2
\right]\right)
$$

Its total integral over all $(x, y)$ is 1 by construction.

But what about a **finite region** $R$? Say, the rectangle
$[a, b]\!\times\![c, d]$? The closed form is

$$
P_R = \int_{a}^{b} \int_{c}^{d} p(x, y)\,dy\,dx
    = [F(b, d) - F(a, d) - F(b, c) + F(a, c)]
$$

where
$F(x, y) = \Phi\left(\frac{x-\mu_x}{\sigma_x}\right)
                 \Phi\left(\frac{y-\mu_y}{\sigma_y}\right)$
and $\Phi$ is the standard normal CDF.

---

## 3 · Monte Carlo Integration

Suppose you can't integrate analytically. Monte Carlo says: sample a ton of
random points in $R$, evaluate $f(x, y)$, average them, and multiply by area.

$$
\iint_{R} f(x, y)\,dx\,dy
\approx \text{area}(R) \cdot \frac{1}{N}\sum_{i=1}^{N} f(x_i, y_i)
$$

where $(x_i, y_i)$ are random uniform samples in $R$.

---

## 4 · Python Demo — Integrating a 2‑D Gaussian Over a Rectangle

```python
# calc-08-multint/mc_gauss_2d.py
import numpy as np
from scipy.stats import norm

# --- parameters for Gaussian
mux, muy = 0.0, 0.0
sigx, sigy = 1.0, 1.0

def pxy(x, y):
    return (1.0/(2*np.pi*sigx*sigy) *
            np.exp(-0.5*((x-mux)**2/sigx**2 + (y-muy)**2/sigy**2)))

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
```

Typical output:

```
Closed‑form prob in box: 0.46607
Monte Carlo estimate  : 0.46601
Absolute error        : 5.91e-05
```

---

## 5 · Why ML Cares: Expected Loss & ELBO

- **Expected loss:** When data is random, loss = average over the data
  distribution, i.e., $\mathbb{E}_\mathbf{x}[L(f(\mathbf{x}))]$.
- **Evidence Lower Bound (ELBO):** Core to variational inference; typically
  estimated with Monte Carlo samples in high‑D.

Monte Carlo = “try points, average, repeat.”

---

## 6 · Exercises

1. **Visualize the 2‑D Gaussian:** Make a filled contour plot of $p(x, y)$ for
   $(x, y) \in [-3, 3]^2$. Mark the integration rectangle.
2. **Vary Region Size:** Compute the Monte Carlo estimate for squares of size 2,
   3, and 4 (i.e., sides \[−L/2, L/2]), compare error to closed form.
3. **Non‑Gaussian Integrand:** Use Monte Carlo to estimate
   $\iint_{R} \tanh(x+y)\,p(x, y)\,dx\,dy$. Compare with the analytic value from
   `scipy.integrate.dblquad` (if feasible).
4. **Convergence Plot:** For $N$ from 100 to 1,000,000, plot the MC estimate and
   error vs. $N$ (on log‑x axis). How fast does the estimate converge?

Put solutions in `calc-08-multint/` and tag `v0.1`.

---

**Next:** _Calculus 9 — Change of Variables & Jacobian Determinants: Pushing
Probability Through Neural Networks._
