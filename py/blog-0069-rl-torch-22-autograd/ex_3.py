import torch

# ### **Exercise 2:** Calculate Gradients for a Multi-variable Function

# - Let $x=1.5$, $y=-2.0$, both with `requires_grad=True`.
# - Define $f(x,y) = 5x^2 + xy + 2y^3$.
# - Compute $f(x, y)$ and call `.backward()`.
# - Print `x.grad` and `y.grad`.
x = torch.tensor(1.5, requires_grad=True)  # Create tensor x with requires_grad=True
y = torch.tensor(-2.0, requires_grad=True)  # Create tensor y with requires_grad=True
f = 5 * x**2 + x * y + 2 * y**3  # Define the function f(x, y)
f.backward(retain_graph=True)  # Compute the gradients
print("Value of f(x, y):", f.item())  # Print the value of f(x, y)
if x.grad is not None:
    print("Gradient at x=1.5 (df/dx):", x.grad.item())  # Print the gradient df/dx
# - Re-run the previous gradient calculation **twice in a row** without zeroing
#   gradients.
# - Observe the value of `.grad` on the second `.backward()`.
# - Now use `.grad.zero_()` after each `.backward()` and verify `.grad` is correct
#   each time.

if y.grad is not None:
    print("Gradient at y=-2.0 (df/dy):", y.grad.item())  # Print the gradient df/dy
# Re-run the backward pass without zeroing gradients
f.backward(retain_graph=True)  # This will accumulate gradients
print("After second backward pass:")
if x.grad is not None:
    print("Gradient at x=1.5 (df/dx) after second backward:", x.grad.item())
if y.grad is not None:
    print("Gradient at y=-2.0 (df/dy) after second backward:", y.grad.item())

# Zero the gradients before the next backward pass
if x.grad is not None:
    x.grad.zero_()
if y.grad is not None:
    y.grad.zero_()
# Re-run the backward pass after zeroing gradients
f.backward()  # This will compute gradients again
print("After zeroing gradients and second backward pass:")
if x.grad is not None:
    print("Gradient at x=1.5 (df/dx) after zeroing:", x.grad.item())
if y.grad is not None:
    print("Gradient at y=-2.0 (df/dy) after zeroing:", y.grad.item())
