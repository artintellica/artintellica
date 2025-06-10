import torch

# ### **Exercise 1:** Mark a Tensor as `requires_grad=True` and Compute Gradients

# - Create a tensor $z = 5.0$ with `requires_grad=True`.
# - Let $f(z) = 3z^2 + 4z$.
# - Compute $f(z)$ and call `.backward()`.
# - Print the gradient `z.grad`.
z = torch.tensor(5.0, requires_grad=True)  # Create tensor with requires_grad=True
f = 3 * z**2 + 4 * z  # Define the function f(z)
f.backward()  # Compute the gradient
print("Value of f(z):", f.item())  # Print the value of f(z)
if z.grad is not None:
    print("Gradient at z=5 (df/dz):", z.grad.item())  # Print the gradient df/dz
