import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(-1.0, requires_grad=True)
f = x**2 + y**3
f.backward()  # Compute gradients for both x and y

print("f(x, y):", f.item())
print(
    "df/dx at (x=2):", x.grad.item() if x.grad is not None else "Error"
)  # Should be 4.0
print(
    "df/dy at (y=-1):", y.grad.item() if y.grad is not None else "Error"
)  # Should be 3*(-1)**2 = 3.0
