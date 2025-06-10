import torch

# ### **Exercise 2:** Calculate Gradients for a Multi-variable Function

# - Let $x=1.5$, $y=-2.0$, both with `requires_grad=True`.
# - Define $f(x,y) = 5x^2 + xy + 2y^3$.
# - Compute $f(x, y)$ and call `.backward()`.
# - Print `x.grad` and `y.grad`.
x = torch.tensor(1.5, requires_grad=True)  # Create tensor x with requires_grad=True
y = torch.tensor(-2.0, requires_grad=True)  # Create tensor y with requires_grad=True
f = 5 * x**2 + x * y + 2 * y**3  # Define the function f(x, y)
f.backward()  # Compute the gradients
print("Value of f(x, y):", f.item())  # Print the value of f(x, y)
if x.grad is not None:
    print("Gradient at x=1.5 (df/dx):", x.grad.item())  # Print the gradient df/dx
