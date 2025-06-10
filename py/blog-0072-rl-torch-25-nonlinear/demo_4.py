import torch
import matplotlib.pyplot as plt

N = 120
torch.manual_seed(0)
x = torch.linspace(-3, 3, N)
# True relationship: cubic (degree 3) with noise
y_true = 0.4 * x**3 - x**2 + 0.5 * x + 2.0
y = y_true + 2.0 * torch.randn(N)


# Helper: build polynomial feature matrix (design matrix) Phi(x), for degree d
def poly_features(x: torch.Tensor, degree: int) -> torch.Tensor:
    return torch.stack([x**i for i in range(degree + 1)], dim=1)  # Shape: (N, degree+1)


degree = 3  # Try cubic first
X_poly = poly_features(x, degree)  # (N, 4)
w = torch.zeros(degree + 1, requires_grad=True)
lr = 0.0002

losses = []
for epoch in range(3000):
    y_pred = X_poly @ w
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad if w.grad is not None else 0
    w.grad.zero_() if w.grad is not None else None
    losses.append(loss.item())
    if epoch % 500 == 0 or epoch == 2999:
        print(f"Epoch {epoch}: loss={loss.item():.3f}")

# Linear
X_lin = poly_features(x, 1)
w_lin = torch.zeros(2, requires_grad=True)
for epoch in range(400):
    y_pred_lin = X_lin @ w_lin
    loss_lin = ((y_pred_lin - y) ** 2).mean()
    loss_lin.backward()
    with torch.no_grad():
        w_lin -= 0.001 * w_lin.grad if w_lin.grad is not None else 0
    w_lin.grad.zero_() if w_lin.grad is not None else None

# Cubic (reuse from above)
with torch.no_grad():
    y_fit_lin = X_lin @ w_lin
    y_fit_poly = X_poly @ w

degrees = [1, 3, 4, 5, 6]
plt.scatter(x, y, alpha=0.2, label="Noisy Data")

colors = ["b", "g", "orange", "r"]
for deg, c in zip(degrees, colors):
    w = torch.zeros(deg+1, requires_grad=True)
    Xp = poly_features(x, deg)
    for epoch in range(800):
        y_pred = Xp @ w
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= 0.001 * w.grad if w.grad is not None else 0
        w.grad.zero_() if w.grad is not None else None
    with torch.no_grad():
        y_fit = Xp @ w
    plt.plot(x, y_fit, color=c, label=f"degree={deg}")

plt.plot(x, y_true, "k--", label="True curve")
plt.xlabel("x"); plt.ylabel("y")
plt.legend(); plt.title("Polynomial Degree and Overfitting")
plt.show()
