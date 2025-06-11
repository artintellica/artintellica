import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)
N = 200
# Two Gaussian blobs
mean0 = torch.tensor([-2.0, 0.0])
mean1 = torch.tensor([2.0, 0.5])
cov = torch.tensor([[1.0, 0.5], [0.5, 1.2]])
L = torch.linalg.cholesky(cov)

X0 = torch.randn(N // 2, 2) @ L.T + mean0
X1 = torch.randn(N // 2, 2) @ L.T + mean1
X = torch.cat([X0, X1], dim=0)
y = torch.cat([torch.zeros(N // 2), torch.ones(N // 2)])


def sigmoid(z: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-z))


# Add bias feature for simplicity: [x1, x2, 1]
X_aug = torch.cat([X, torch.ones(N, 1)], dim=1)  # shape (N, 3)
w = torch.zeros(3, requires_grad=True)
lr = 0.05

losses = []
for epoch in range(1500):
    z = X_aug @ w  # Linear
    p = sigmoid(z)  # Probabilities
    # Numerical stabilization: clamp p
    eps = 1e-8
    p = p.clamp(eps, 1 - eps)
    bce = (-y * torch.log(p) - (1 - y) * torch.log(1 - p)).mean()
    bce.backward()
    with torch.no_grad():
        w -= lr * w.grad if w.grad is not None else 0
    w.grad.zero_() if w.grad is not None else None
    losses.append(bce.item())
    if epoch % 300 == 0 or epoch == 1499:
        print(f"Epoch {epoch}: BCE loss={bce.item():.3f}")

w2 = torch.zeros(3, requires_grad=True)
optimizer = torch.optim.SGD([w2], lr=0.05)
losses2 = []
for epoch in range(1500):
    z = X_aug @ w2
    p = sigmoid(z)
    p = p.clamp(1e-8, 1 - 1e-8)
    bce = torch.nn.functional.binary_cross_entropy(p, y)
    bce.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses2.append(bce.item())

print("Final weights (manual):", w.detach().numpy())
print("Final weights (optimizer):", w2.detach().numpy())


with torch.no_grad():
    # Grid for decision boundary
    x1g, x2g = torch.meshgrid(
        torch.linspace(-6, 6, 100), torch.linspace(-4, 5, 100), indexing="ij"
    )
    Xg = torch.stack([x1g.reshape(-1), x2g.reshape(-1), torch.ones(100 * 100)], dim=1)
    p_grid = sigmoid(Xg @ w2).reshape(100, 100)

    # Predictions for accuracy
    preds = (sigmoid(X_aug @ w2) > 0.5).float()
    acc = (preds == y).float().mean().item()

plt.contourf(
    x1g, x2g, p_grid, levels=[0, 0.5, 1], colors=["lightblue", "salmon"], alpha=0.2
)
plt.scatter(X0[:, 0], X0[:, 1], color="b", alpha=0.5, label="Class 0")
plt.scatter(X1[:, 0], X1[:, 1], color="r", alpha=0.5, label="Class 1")
plt.title(f"Decision Boundary (Accuracy: {acc*100:.1f}%)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()
