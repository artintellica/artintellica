#!/usr/bin/env python3
"""
exercise_3_posenc_link.py
-------------------------------------------------
Truncate  e^{iθ}  to its first two non‑zero Taylor terms:

      e^{iθ}  ≈  1 + iθ            (O(θ²))

Split real / imag parts to recover the linear
approximations of  cos θ  and  sin θ, and connect
them to the sine / cosine pairs used in the original
Transformer positional encoding.
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. θ grid and exact complex exponential
# ------------------------------------------------
theta = np.linspace(-np.pi, np.pi, 601)  # [-π, π]
exp_i = np.exp(1j * theta)  # e^{iθ} = cosθ + i sinθ

cos_true = exp_i.real
sin_true = exp_i.imag

# ------------------------------------------------
# 2. First‑order truncation: 1 + iθ
# ------------------------------------------------
trunc = 1 + 1j * theta
cos_lin = trunc.real  # = 1
sin_lin = trunc.imag  # = θ

# errors for later annotation
err_cos = np.abs(cos_true - cos_lin)
err_sin = np.abs(sin_true - sin_lin)

print(f"Max |cosθ - 1|  on [-π,π] : {err_cos.max():.3f}")
print(f"Max |sinθ - θ|  on [-π,π] : {err_sin.max():.3f}")

# ------------------------------------------------
# 3. Plot comparison
# ------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

# --- cosine -----------------------------------------------------------
ax[0].plot(theta, cos_true, label="cos θ", lw=2)
ax[0].plot(theta, cos_lin, "--", label="1 (trunc)", lw=1.5)
ax[0].set_title("Real part")
ax[0].set_xlabel("θ")
ax[0].set_ylabel("value")
ax[0].legend()

# --- sine -------------------------------------------------------------
ax[1].plot(theta, sin_true, label="sin θ", lw=2)
ax[1].plot(theta, sin_lin, "--", label="θ (trunc)", lw=1.5)
ax[1].set_title("Imag part")
ax[1].set_xlabel("θ")
ax[1].legend()

plt.suptitle("Truncating  e^{iθ}  →  1 + iθ")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ------------------------------------------------
# 4. Positional‑encoding link (text output)
# ------------------------------------------------
print(
    """
Transformer PE uses   [sin(ω·pos), cos(ω·pos)]  pairs.

Near zero,  sin(ω·pos) ≈ ω·pos   and   cos(ω·pos) ≈ 1,
exactly the real/imag parts of  1 + i(ω·pos).

Thus the first‑order Taylor truncation explains why
the PE starts as a *linear* ramp (sin) combined with
a constant offset (cos), before higher‑order terms
introduce curvature farther from the origin.
"""
)
