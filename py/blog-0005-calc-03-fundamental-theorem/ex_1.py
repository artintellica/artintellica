
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import integrate

a, b = -1.0, 1.0
exact = norm.cdf(b) - norm.cdf(a)

ns         = list(range(201, 2002, 200))     # 201, 401, …, 2001
err_trap   = []
err_simp   = []

for n in ns:
    xs  = np.linspace(a, b, n)
    pdf = norm.pdf(xs)
    err_trap.append(abs(np.trapz(pdf, xs)           - exact))
    err_simp.append(abs(integrate.simpson(pdf, xs)  - exact))

plt.figure(figsize=(6,4))
plt.loglog(ns, err_trap, "o-", label="Trapezoid")
plt.loglog(ns, err_simp, "s-", label="Simpson")
plt.xlabel("Number of samples (N)")
plt.ylabel("Absolute error")
plt.title("Accuracy vs. N  for ∫_{-1}^{1} φ(t) dt")
plt.legend()
plt.tight_layout()
plt.show()
