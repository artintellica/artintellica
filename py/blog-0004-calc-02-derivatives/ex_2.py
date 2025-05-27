"""
Check ∂/∂x tanh(x) with autograd, finite diff, analytic formula 1 − tanh²x.
"""

import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)  # higher accuracy

# 1. sample some points
x = torch.linspace(-3, 3, 61, requires_grad=True)  # 61 points in [−3,3]
y = torch.tanh(x)

# 2. autograd gradient
(grad_auto,) = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))

# 3. finite difference (centered)
h = 1e-4
with torch.no_grad():
    y_plus = torch.tanh(x + h)
    y_minus = torch.tanh(x - h)
grad_fd = (y_plus - y_minus) / (2 * h)

# 4. analytic gradient
grad_true = 1.0 - torch.tanh(x) ** 2

# 5. print max errors
err_auto = torch.max(torch.abs(grad_auto - grad_true)).item()
err_fd = torch.max(torch.abs(grad_fd - grad_true)).item()
print(f"max |autograd − true|   = {err_auto:.3e}")
print(f"max |finite‑diff − true| = {err_fd:.3e}")

# 6. quick visual check
x_np = x.detach().numpy()
plt.figure(figsize=(6, 4))
plt.plot(x_np, grad_true.detach().numpy(), label="analytic 1−tanh²x", linewidth=2)
plt.plot(x_np, grad_auto.detach().numpy(), "--", label="autograd", linewidth=1)
plt.plot(x_np, grad_fd.detach().numpy(), ":", label="finite diff", linewidth=1)
plt.legend()
plt.xlabel("x")
plt.ylabel("derivative")
plt.title("Derivative of tanh(x) — three methods")
plt.tight_layout()
plt.show()
