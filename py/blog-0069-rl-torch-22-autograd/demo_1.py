import torch

x = torch.tensor(3.0, requires_grad=True)
f = x**2
f.backward()  # Compute the gradient ∂f/∂x at x=3

print("Value of f(x):", f.item())
print(
    "Gradient at x=3 (df/dx):",
    x.grad.item() if x.grad is not None else "No gradient computed",
)
