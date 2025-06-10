import torch

# Let’s use f(x) = x^2 at x=4.0, df/dx should be 8.0
x = torch.tensor(4.0, requires_grad=True)
f = x**2
f.backward()
if x.grad is not None:
    print("Value of f(x):", f.item())

# Manual calculation
manual_grad = 2 * x.item()  # 2 * 4.0 = 8.0
print("Manual grad:", manual_grad)
