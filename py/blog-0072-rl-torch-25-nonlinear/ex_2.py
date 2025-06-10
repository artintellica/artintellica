import torch
import matplotlib.pyplot as plt


def poly_features(x: torch.Tensor, d: int) -> torch.Tensor:
    return torch.stack([x**i for i in range(d + 1)], dim=1)


N = 150
x = torch.linspace(-2.5, 2.5, N)
y_true = -0.5 * x**3 + 1.2 * x**2 - 0.7 * x + 4
torch.manual_seed(0)
y = y_true + 1.2 * torch.randn(N)

degree = 3
Xp = poly_features(x, degree)
w = torch.zeros(degree + 1, requires_grad=True)
losses = []
for epoch in range(2500):
    y_pred = Xp @ w
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= 0.001 * w.grad if w.grad is not None else 0
    w.grad.zero_() if w.grad is not None else None
    losses.append(loss.item())
plt.plot(losses)
plt.title("Polynomial Regression Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
