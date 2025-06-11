import torch
import numpy as np
import matplotlib.pyplot as plt


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
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="r", label="Class 0", alpha=0.6)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="b", label="Class 1", alpha=0.6)
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Two Moons Data")
plt.show()
