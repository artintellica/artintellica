# demo_limits.py
import numpy as np


def limit_tester(f, c, L, eps=1e-4):
    # naive sweep for a suitable δ
    for delta_exp in range(-1, -8, -1):  # 0.1, 0.01, …, 1e‑7
        δ = 10.0**delta_exp
        xs = np.linspace(c - δ, c + δ, 1001)
        xs = xs[xs != c]
        if np.all(np.abs(f(xs) - L) < eps):
            return δ
    return None


f = lambda x: np.sin(x) / x
δ_found = limit_tester(f, 0.0, 1.0, eps=1e-5)
print(f"Found δ = {δ_found} for ε = 1e-5")
