import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


def make_moons(n_samples=200, noise=0.1):
    # Generate two interleaving half circles ("moons"), similar to sklearn.datasets.make_moons
    n = n_samples // 2
    theta = np.pi * np.random.rand(n)
    x0 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    x1 = np.stack([1 - np.cos(theta), 1 - np.sin(theta)], axis=1) + np.array(
        [0.6, -0.4]
    )
    X = np.vstack([x0, x1])
    X += noise * np.random.randn(*X.shape)
    y = np.hstack([np.zeros(n), np.ones(n)])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


torch.manual_seed(42)
np.random.seed(42)
X, y = make_moons(200, noise=0.2)

model = nn.Sequential(
    nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 2)  # 2 outputs = logits for 2 classes
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()

# Training loop
losses = []
for epoch in range(200):
    logits = model(X)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 40 == 0 or epoch == 199:
        print(f"Epoch {epoch}: loss = {loss.item():.3f}")

with torch.no_grad():
    # Create a grid of points covering the data
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 200),
        np.linspace(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, 200),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    logits_grid = model(grid_tensor)
    probas = torch.softmax(logits_grid, dim=1)
    preds = probas.argmax(dim=1).cpu().numpy().reshape(xx.shape)

plt.contourf(xx, yy, preds, alpha=0.2, cmap="coolwarm", levels=2)
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="r", label="Class 0", alpha=0.7)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="b", label="Class 1", alpha=0.7)
plt.title("Learned Decision Boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()
