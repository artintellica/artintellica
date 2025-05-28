# calc-06-gradients/autograd_xy2.py
import torch

torch.set_default_dtype(torch.float64)


def f(xy):
    x, y = xy
    return x**2 + y**2


point = torch.tensor([1.5, -0.8], requires_grad=True)
val = f(point)
val.backward()

print(f"f({point.tolist()}) = {val.item():.3f}")
print("∇f =", point.grad.tolist())  # should be [2*1.5, 2*(-0.8)]
