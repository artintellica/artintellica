import torch

# Define a non-positive definite matrix D
D = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
x = torch.tensor([1.0, 1.0], requires_grad=True)

# Quadratic form as loss: x^T D x
initial_loss = torch.matmul(x, torch.matmul(D, x))
print("Initial loss:", initial_loss.item())
print("Initial x:", x.data)

# Gradient descent
optimizer = torch.optim.SGD([x], lr=0.1)
for step in range(10):
    optimizer.zero_grad()
    loss = torch.matmul(x, torch.matmul(D, x))
    loss.backward()
    optimizer.step()
    print(f"Step {step+1} - Loss: {loss.item():.4f}, x: {x.data}")

print("\nFinal loss:", loss.item())
print("Final x:", x.data)

# Observations on behavior compared to the positive definite case:
# Unlike the positive definite case (e.g., with matrix A = [[4, 1], [1, 3]]), where the loss decreases
# and converges to a minimum at x = [0, 0], here with a non-positive definite matrix D, the behavior
# is different. The matrix D has eigenvalues 1 and -1, indicating a saddle point rather than a minimum.
# The loss x^T D x = x1^2 - x2^2 does not have a global minimum; it can become arbitrarily negative
# as x2 increases. In this run, gradient descent updates x1 towards 0 (since the gradient for x1 is 2*x1),
# reducing the positive contribution, while x2 grows larger (since the gradient for x2 is -2*x2, pushing
# it away from 0), making the loss more negative. This contrasts with the positive definite case, where
# the loss is always positive and converges to a minimum.
