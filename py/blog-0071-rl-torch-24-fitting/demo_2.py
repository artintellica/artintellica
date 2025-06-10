import torch
import matplotlib.pyplot as plt

# True parameters
w_true = 2.5
b_true = -1.7
N = 120
torch.manual_seed(42)

# Generate random x and noisy y
x = torch.linspace(-3, 3, N)
y = w_true * x + b_true + 0.9 * torch.randn(N)

# Initialize parameters
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.04

losses = []
for epoch in range(80):
    y_pred = w * x + b  # Linear model
    loss = ((y_pred - y) ** 2).mean()  # MSE
    loss.backward()
    with torch.no_grad():
        if w.grad is not None:
            w -= lr * w.grad
        if b.grad is not None:
            b -= lr * b.grad
    if w.grad is not None:
        w.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()
    losses.append(loss.item())
    if epoch % 20 == 0 or epoch == 79:
        print(
            f"Epoch {epoch:2d}: w={w.item():.2f}, b={b.item():.2f}, loss={loss.item():.3f}"
        )

plt.plot(losses)
plt.title("Training Loss (From Scratch)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
