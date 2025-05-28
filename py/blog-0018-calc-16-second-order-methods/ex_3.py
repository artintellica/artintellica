"""
exercise_3_hvp_vs_fullH.py
-------------------------------------------------
Hessian‑vector product versus explicit Hessian
for a random 20‑D quadratic f(x)=½ xᵀA x.
"""

import time
import torch
import numpy as np

torch.set_default_dtype(torch.float64)

# ---------- make random positive‑definite A ----------
n = 20
rng = np.random.default_rng(0)
Q = torch.tensor(rng.standard_normal((n, n)))
A = Q.T @ Q + n * torch.eye(n)  # SPD


# function, point, random v
def f(x):
    return 0.5 * x @ A @ x


x0 = torch.randn(n, requires_grad=True)
v = torch.randn(n)

# ---------- Hessian‑vector product (autograd trick) ----------
t0 = time.time()
g = torch.autograd.grad(f(x0), x0, create_graph=True)[0]
hvp = torch.autograd.grad(g @ v, x0)[0]
t_hvp = time.time() - t0
print(f"Hv by autograd   : {t_hvp*1e3:.2f} ms")

# ---------- full Hessian then multiply ----------
t1 = time.time()
H = torch.autograd.functional.hessian(f, x0)  # full n×n matrix
full = H @ v
t_full = time.time() - t1
print(f"Full Hessian*vec : {t_full*1e3:.2f} ms")

# ---------- check correctness ----------
err = torch.norm(hvp - full)
print(f"‖difference‖ = {err:.3e}")
