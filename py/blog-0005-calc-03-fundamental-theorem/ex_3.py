# calc-03-ftc/parameter_gradient.py

import torch
import math

torch.set_default_dtype(torch.float64)  # high accuracy

# ---------- helpers ----------
SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


def phi(x):  # standard‑normal pdf
    return torch.exp(-0.5 * x**2) / SQRT_TWO_PI


def G(mu, n=8001):
    """
    Differentiable quadrature of ∫_{-2}^{2} N(t; mu, 1) dt
    using a fixed trapezoid grid with n points.
    mu may be a scalar or a 1‑D tensor of shape [k].
    """
    t = torch.linspace(-2.0, 2.0, n, device=mu.device)  # [n]
    pdf = phi(t.unsqueeze(0) - mu.unsqueeze(1))  # [k,n]
    # integrate along last dim
    return torch.trapz(pdf, t, dim=-1)  # [k]


# ---------- test values ----------
mu = torch.tensor([-1.0, 0.0, 0.5, 2.0], requires_grad=True)  # [k]

# ---------- forward + backward ----------
G_vals = G(mu)  # [k]
G_vals.sum().backward()  # accumulate and back‑prop once
grad_auto = mu.grad  # [k]

# ---------- analytic gradient ----------
grad_true = -(phi(2 - mu.detach()) - phi(-2 - mu.detach()))

# ---------- report ----------
print(
    f"{'mu':>6} | {'G(mu)':>10} | {'autograd dG/dmu':>18} | {'analytic':>10} | abs err"
)
print("-" * 70)
for m, gval, ga, gt in zip(mu, G_vals, grad_auto, grad_true):
    err = abs(ga - gt)
    print(f"{float(m):6.2f} | {gval:10.6f} | {ga:18.12f} | {gt:10.12f} | {err:.2e}")
