# calc-03-ftc/ftc_autograd.py
import torch
import math

torch.set_default_dtype(torch.float64)


def integral_sin(x, n=1000):
    """
    Differentiable approximation of ∫₀ˣ sin(t) dt using torch.trapz.
    """
    t_base = torch.linspace(0.0, 1.0, n, device=x.device)  # fixed grid [0,1]
    t = t_base * x  # scale to [0,x]
    y = torch.sin(t)
    return torch.trapz(y, t)  # area under curve


x = torch.tensor(1.2, requires_grad=True)
F = integral_sin(x)
F.backward()  # ∂F/∂x via autograd

print(f"Integral F(1.2) ≈ {F.item():.8f}")
print(f"autograd dF/dx  ≈ {x.grad.item():.8f}")
print(f"analytic sin(1.2)= {math.sin(1.2):.8f}")
