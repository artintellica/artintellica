import torch

# Define a positive definite matrix A
A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
x = torch.tensor([1.0, 1.0], requires_grad=True)

# Quadratic form as loss: x^T A x
loss = torch.matmul(x, torch.matmul(A, x))
print("Initial loss:", loss.item())

# Gradient descent
optimizer = torch.optim.SGD([x], lr=0.1)
for _ in range(10):
    optimizer.zero_grad()
    loss = torch.matmul(x, torch.matmul(A, x))
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}, x: {x.data}")

print("Final x (should be near [0, 0]):", x.data)
