from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch

def generate_moons(n_samples: int = 200, noise: float = 0.20) -> Tuple[torch.Tensor, torch.Tensor]:
    X_np, y_np = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    return X, y

X, y = generate_moons()
plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap="coolwarm", s=40)
plt.title("Synthetic Nonlinear Dataset: Two Moons")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

