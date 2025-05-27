import torch
import math

# sample test points (float64 for more accuracy)
xs = torch.tensor(
    [-math.pi, -math.pi / 2, -math.pi / 4, 0.0, math.pi / 4, math.pi / 2, math.pi],
    dtype=torch.float64,
    requires_grad=True,
)

# autograd gradient of sin(x)  ——>  should be cos(x)
y = torch.sin(xs)
grads_autograd = torch.autograd.grad(
    outputs=y, inputs=xs, grad_outputs=torch.ones_like(y)
)[0]

# finite‑difference estimate: (f(x+h) – f(x–h)) / (2h)
h = 1e-6  # small step; double precision
with torch.no_grad():
    f_plus = torch.sin(xs + h)
    f_minus = torch.sin(xs - h)
grads_fd = (f_plus - f_minus) / (2 * h)

# analytic truth for comparison
grads_true = torch.cos(xs)

# report
print(f"{'x':>12} | {'autograd':>12} | {'finite diff':>12} | {'cos(x)':>12}")
print("-" * 55)
for x, g_auto, g_fd, g_true in zip(xs, grads_autograd, grads_fd, grads_true):
    print(
        f"{float(x):12.8f} | {float(g_auto):12.8f} | "
        f"{float(g_fd):12.8f} | {float(g_true):12.8f}"
    )
