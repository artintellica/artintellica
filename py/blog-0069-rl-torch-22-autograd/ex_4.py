import torch

### **Exercise 4:** Manually Verify PyTorch’s Computed Gradients for a Small Example

# - Use $f(x) = 7x^2$ at $x = 3.0$.
# - Compute the gradient using autograd and by hand.
# - Ensure the answers match.
x = torch.tensor(3.0, requires_grad=True)  # Create tensor x with requires_grad=True
f = 7 * x**2  # Define the function f(x)
f.backward()  # Compute the gradient using autograd
# Manually compute the gradient
manual_gradient = 14 * x.item()  # df/dx = 14 * x at x = 3.0
print("Value of f(x):", f.item())  # Print the value of f(x)
if x.grad is not None:
    print("Gradient at x=3.0 (df/dx):", x.grad.item())  # Print the gradient df/dx
print("Manual gradient:", manual_gradient)  # Print the manually computed gradient
