#!/usr/bin/env python3
"""
exercise_2_heat_kernel_smoothing.py
-------------------------------------------------
Smooth a noisy sine wave by exact heat‑equation evolution
using the FFT (periodic boundary, no CFL issues).
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# grid & noisy data
# ------------------------------------------------------------------
N = 300
x = np.linspace(0, 1, N, endpoint=False)
dx = x[1] - x[0]

rng = np.random.default_rng(42)
f0 = np.sin(2 * np.pi * x) + 0.3 * rng.standard_normal(N)

# ------------------------------------------------------------------
# FFT helpers
# ------------------------------------------------------------------
k = np.fft.fftfreq(N, d=dx)  # cycles per unit length
k2 = (2 * np.pi * k) ** 2  # (2πk)^2


def heat_evolve(f, t):
    """Return f(x,t) = e^{t Δ} f0 via FFT (periodic BC)."""
    F_hat = np.fft.fft(f)
    F_hat *= np.exp(-k2 * t)  # spectral decay
    return np.real(np.fft.ifft(F_hat))


# choose three smoothing times
taus = [0.0, 5e-4, 2e-3, 1e-2]  # 0 = original
styles = ["--", "-", "-", "-"]

plt.figure(figsize=(9, 6))
for tau, style in zip(taus, styles):
    f_tau = heat_evolve(f0, tau)
    label = f"t = {tau}" if tau > 0 else "initial"
    plt.plot(x, f_tau, style, lw=2, label=label)

plt.title("Exact heat‑kernel smoothing (FFT)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.tight_layout()
plt.show()
