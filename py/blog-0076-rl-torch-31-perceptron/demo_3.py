import torch
import numpy as np
import matplotlib.pyplot as plt


def perceptron_step(
    x: torch.Tensor, y: float, w: torch.Tensor, b: float, lr: float = 1.0
) -> tuple[torch.Tensor, float]:
    """
    x: input vector (d,)
    y: true label (0 or 1)
    w: weight vector (d,)
    b: bias (scalar)
    Returns updated w, b
    """
    # Prediction: if w^T x + b > 0: 1 else 0
    z: float = torch.dot(w, x).item() + b
    y_pred: float = 1.0 if z > 0 else 0.0
    if y != y_pred:
        # Update rule
        w = w + lr * (y - y_pred) * x
        b = b + lr * (y - y_pred)
    return w, b


# Generate simple separable data
np.random.seed(0)
N: int = 40
X0: np.ndarray = np.random.randn(N, 2) + np.array([2, 2])
X1: np.ndarray = np.random.randn(N, 2) + np.array([-2, -2])
X: np.ndarray = np.concatenate([X0, X1])
y: np.ndarray = np.concatenate([np.ones(N), np.zeros(N)])

X_t: torch.Tensor = torch.tensor(X, dtype=torch.float32)
y_t: torch.Tensor = torch.tensor(y, dtype=torch.float32)
w: torch.Tensor = torch.zeros(2)
b: float = 0.0

epochs: int = 12
boundary_history: list[tuple[torch.Tensor, float]] = []

for epoch in range(epochs):
    for i in range(len(X)):
        w, b = perceptron_step(X_t[i], y_t[i].item(), w, b, lr=0.7)
    boundary_history.append((w.clone(), b))

print("Final weights:", w, "Final bias:", b)


def plot_perceptron_decision(
    X: np.ndarray, y: np.ndarray, boundary_history: list[tuple[torch.Tensor, float]]
) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="orange", label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", label="Class 1")
    x_vals: np.ndarray = np.array(plt.gca().get_xlim())
    for i, (w, b) in enumerate(boundary_history):
        # Line: w1*x + w2*y + b = 0 => y = (-w1*x - b)/w2
        if w[1].abs() > 1e-6:
            y_vals = (-w[0].item() * x_vals - b) / w[1].item()
            plt.plot(
                x_vals,
                y_vals,
                alpha=0.3 + 0.7 * i / len(boundary_history),
                label=f"Epoch {i+1}" if i == len(boundary_history) - 1 else None,
            )
    plt.legend()
    plt.title("Perceptron Decision Boundary Evolution")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


plot_perceptron_decision(X, y, boundary_history)
