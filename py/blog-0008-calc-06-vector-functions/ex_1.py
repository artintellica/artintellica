"""
exercise_1_grad_checker.py
-------------------------------------------------
Gradient checker for

    g(x, y) = sin(x) + cos(y)

Analytic gradient:
    ∇g = [ cos(x),  -sin(y) ]

We generate N random points in [-π, π]², compute both
analytic and autograd gradients, and assert the maximum
absolute difference is below 1e‑8.
"""

import torch
import math
import random

torch.set_default_dtype(torch.float64)


# ------------------------------------------------
# 1. analytic gradient function
# ------------------------------------------------
def grad_g_analytic(xy):
    x, y = xy
    return torch.stack([torch.cos(x), -torch.sin(y)])


# ------------------------------------------------
# 2. autograd wrapper
# ------------------------------------------------
def grad_g_autograd(xy):
    xy = xy.clone().detach().requires_grad_(True)
    g = torch.sin(xy[0]) + torch.cos(xy[1])
    g.backward()
    return xy.grad


# ------------------------------------------------
# 3. run gradient check
# ------------------------------------------------
N = 10
max_err = 0.0

for i in range(N):
    # random point in [-π, π]^2
    point = torch.tensor(
        [random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi)],
        dtype=torch.float64,
    )

    grad_an = grad_g_analytic(point)
    grad_ad = grad_g_autograd(point)
    err = torch.max(torch.abs(grad_an - grad_ad)).item()
    max_err = max(max_err, err)

    print(f"point {i:2}: (x,y)=({point[0]:+.3f}, {point[1]:+.3f})  " f"err={err:.2e}")

print("\nMax absolute error across all points:", f"{max_err:.2e}")
assert max_err < 1e-8, "Gradient check failed!"
print("✓  All gradients match to < 1e‑8")
