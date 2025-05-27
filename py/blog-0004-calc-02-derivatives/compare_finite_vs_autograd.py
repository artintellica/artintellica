# calc-02-derivatives/compare_finite_vs_autograd.py
import torch
import math
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# sample many x for a smooth curve
x = torch.linspace(-math.pi, math.pi, 400, requires_grad=True)
y = torch.sin(x)

# autograd derivative (should be cos(x))
(grads_auto,) = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))

# finite difference derivative
h = 1e-4
with torch.no_grad():
    y_plus = torch.sin(x + h)
    y_minus = torch.sin(x - h)
grads_fd = (y_plus - y_minus) / (2 * h)

# ground truth
grads_true = torch.cos(x)

# --- Plot ----------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(x.detach(), grads_true.detach(), label="cos(x)  (true)", linewidth=2)
plt.plot(x.detach(), grads_auto.detach(), "--", label="autograd", linewidth=1)
plt.plot(x.detach(), grads_fd.detach(), ":", label="finite diff", linewidth=1)
plt.legend()
plt.xlabel("x")
plt.ylabel("derivative")
plt.title("sin'(x) via three methods")
plt.tight_layout()
plt.show()
