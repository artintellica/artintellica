# calc-03-ftc/normal_area.py
import numpy as np
from scipy.stats import norm
from scipy import integrate

a, b = -1.0, 1.0
xs = np.linspace(a, b, 2001)  # high resolution grid
pdf = norm.pdf(xs)

area_trap = np.trapz(pdf, xs)  # 1️⃣ Trapezoid
area_simp = integrate.simpson(pdf, xs)  # 2️⃣ Simpson
area_true = norm.cdf(b) - norm.cdf(a)  # 3️⃣ Exact

print(f"Trapezoid  ≈ {area_trap:.8f}")
print(f"Simpson    ≈ {area_simp:.8f}")
print(f"Exact      = {area_true:.8f}")
