"""
exercise_1_chain_rule.py
-------------------------------------------------
For  h(x, y) = tanh(a x + b y), compute the Jacobian

    [∂h/∂x, ∂h/∂y] = (1 - tanh²(ax+by)) * [a, b]

Check analytic vs autograd at random (x, y), (a, b).
"""

import torch
import random

torch.set_default_dtype(torch.float64)


def h(xy, ab):
    x, y = xy
    a, b = ab
    return torch.tanh(a * x + b * y)


def analytic_jacobian(xy, ab):
    x, y = xy
    a, b = ab
    z = a * x + b * y
    sech2 = 1 - torch.tanh(z) ** 2
    return torch.stack([sech2 * a, sech2 * b])


def autograd_jacobian(xy, ab):
    xy = xy.clone().detach().requires_grad_(True)
    ab = ab.clone().detach().requires_grad_(False)
    hval = h(xy, ab)
    hval.backward()
    return xy.grad


N = 5
max_err = 0.0
for i in range(N):
    x, y = [random.uniform(-2, 2) for _ in range(2)]
    a, b = [random.uniform(-2, 2) for _ in range(2)]
    xy = torch.tensor([x, y], dtype=torch.float64)
    ab = torch.tensor([a, b], dtype=torch.float64)
    jac_an = analytic_jacobian(xy, ab)
    jac_ad = autograd_jacobian(xy, ab)
    err = torch.max(torch.abs(jac_an - jac_ad)).item()
    max_err = max(max_err, err)
    print(
        f"({x: .2f},{y: .2f}), a={a:.2f}, b={b:.2f} | analytic: {jac_an.tolist()} | autograd: {jac_ad.tolist()} | err={err:.2e}"
    )

print("\nMax absolute error across all points:", f"{max_err:.2e}")
assert max_err < 1e-8, "Chain rule check failed!"
print("✓  Chain rule analytic and autograd Jacobian match to < 1e‑8")
