import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-3.0, requires_grad=True)
c = a * b + a**2
d = torch.tanh(c)
d.backward()
print(f"d = {d.item():.5f}, ∂d/∂a = {a.grad.item():.5f}, ∂d/∂b = {b.grad.item():.5f}")
