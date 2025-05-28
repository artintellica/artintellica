#!/usr/bin/env python3
"""
exercise_3_weight_decay_connection.py
-------------------------------------------------
Demonstrate that for a constant function  f(x)=c
the smoothness regulariser J = ∫ (f')² dx is zero
and its functional gradient −2 f''(x) is identically zero,
mirroring ordinary L2 weight‑decay where the optimum is at w=0.
"""

import numpy as np

# ------------------------------------------------------------------
# constant function on [0,1]
# ------------------------------------------------------------------
N = 200
x = np.linspace(0, 1, N)
dx = x[1] - x[0]
c = 3.7  # any constant value
f = np.full_like(x, c)

# ------------------------------------------------------------------
# numerical first derivative  f'
# ------------------------------------------------------------------
dfdx = np.zeros_like(f)
dfdx[1:-1] = (f[2:] - f[:-2]) / (2 * dx)  # central diff
dfdx[0] = (f[1] - f[0]) / dx
dfdx[-1] = (f[-1] - f[-2]) / dx

# ------------------------------------------------------------------
# regulariser  J = ∫ (f')² dx
# ------------------------------------------------------------------
J = np.sum(dfdx**2) * dx

# ------------------------------------------------------------------
# functional gradient  −2 f''(x)
# ------------------------------------------------------------------
d2f = np.zeros_like(f)  # because f is constant
grad = -2 * d2f

print(f"Constant c = {c}")
print(f"Max |f'| (should be 0)      : {np.max(np.abs(dfdx)):.3e}")
print(f"Smoothness J = ∫ (f')² dx   : {J:.3e}")
print(f"Max |functional grad| (2f''): {np.max(np.abs(grad)):.3e}")
