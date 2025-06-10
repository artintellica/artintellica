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
# plt.plot(losses)
# plt.title("Polynomial Regression Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()

Xl = poly_features(x, 1)
wl = torch.zeros(2, requires_grad=True)
for epoch in range(800):
    y_pred_lin = Xl @ wl
    loss_lin = ((y_pred_lin - y) ** 2).mean()
    loss_lin.backward()
    with torch.no_grad():
        wl -= 0.001 * wl.grad if wl.grad is not None else 0
    wl.grad.zero_() if wl.grad is not None else None
with torch.no_grad():
    y_fit_lin = Xl @ wl
    y_fit_poly = Xp @ w
# plt.scatter(x, y, alpha=0.3)
# plt.plot(x, y_true, "k--", label="True function")
# plt.plot(x, y_fit_lin, "b-", label="Linear fit")
# plt.plot(x, y_fit_poly, "r-", label="Cubic fit")
# plt.legend()
# plt.show()

degrees = [1, 2, 3, 4, 5, 6, 7]
colors = ['b', 'g', 'orange', 'r', 'purple', 'brown', 'pink']
plt.scatter(x, y, alpha=0.2, label="Noisy data")
for deg, col in zip(degrees, colors):
    w = torch.zeros(deg + 1, requires_grad=True)
    Xd = poly_features(x, deg)
    for epoch in range(1000):
        y_pred = Xd @ w
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= 0.0005 * w.grad if w.grad is not None else 0
        w.grad.zero_() if w.grad is not None else None
    with torch.no_grad():
        y_fit = Xd @ w
    plt.plot(x, y_fit, color=col, label=f"d={deg}")
plt.plot(x, y_true, "k--", label="True")
plt.legend(); plt.title("Overfitting Example"); plt.show()
