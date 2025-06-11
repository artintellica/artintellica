import torch
import numpy as np
import matplotlib.pyplot as plt


# def perceptron_step(
#     x: torch.Tensor, y: float, w: torch.Tensor, b: float, lr: float = 1.0
# ) -> tuple[torch.Tensor, float]:
#     """
#     x: input vector (d,)
#     y: true label (0 or 1)
#     w: weight vector (d,)
#     b: bias (scalar)
#     Returns updated w, b
#     """
#     # Prediction: if w^T x + b > 0: 1 else 0
#     z: float = torch.dot(w, x).item() + b
#     y_pred: float = 1.0 if z > 0 else 0.0
#     if y != y_pred:
#         # Update rule
#         w = w + lr * (y - y_pred) * x
#         b = b + lr * (y - y_pred)
#     return w, b


# def plot_perceptron_decision(
#     X: np.ndarray, y: np.ndarray, boundary_history: list[tuple[torch.Tensor, float]]
# ) -> None:
#     plt.figure(figsize=(8, 5))
#     plt.scatter(X[y == 0, 0], X[y == 0, 1], color="orange", label="Class 0")
#     plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", label="Class 1")
#     x_vals: np.ndarray = np.array(plt.gca().get_xlim())
#     for i, (w, b) in enumerate(boundary_history):
#         # Line: w1*x + w2*y + b = 0 => y = (-w1*x - b)/w2
#         if w[1].abs() > 1e-6:
#             y_vals = (-w[0].item() * x_vals - b) / w[1].item()
#             plt.plot(
#                 x_vals,
#                 y_vals,
#                 alpha=0.3 + 0.7 * i / len(boundary_history),
#                 label=f"Epoch {i+1}" if i == len(boundary_history) - 1 else None,
#             )
#     plt.legend()
#     plt.title("Perceptron Decision Boundary Evolution")
#     plt.xlabel("x1")
#     plt.ylabel("x2")
#     plt.show()


# # Make non-separable data (add noise and overlap)
# np.random.seed(3)
# N: int = 40
# X0_ns: np.ndarray = np.random.randn(N, 2) + np.array([1, 1])
# X1_ns: np.ndarray = np.random.randn(N, 2) + np.array([2, 2])
# X_ns: np.ndarray = np.concatenate([X0_ns, X1_ns])
# y_ns: np.ndarray = np.concatenate([np.zeros(N), np.ones(N)])

# X_t_ns: torch.Tensor = torch.tensor(X_ns, dtype=torch.float32)
# y_t_ns: torch.Tensor = torch.tensor(y_ns, dtype=torch.float32)
# w_ns: torch.Tensor = torch.zeros(2)
# b_ns: float = 0.0
# boundary_history_ns: list[tuple[torch.Tensor, float]] = []

# for epoch in range(12):
#     for i in range(len(X_ns)):
#         w_ns, b_ns = perceptron_step(X_t_ns[i], y_t_ns[i].item(), w_ns, b_ns, lr=0.7)
#     boundary_history_ns.append((w_ns.clone(), b_ns))

# plot_perceptron_decision(X_ns, y_ns, boundary_history_ns)
# print("Final weights (non-separable):", w_ns, "Final bias:", b_ns)

# ### **Exercise 1:** Implement the Perceptron Update Rule for a Binary Task

# - Write a function or code block that updates weights $w$ and bias $b$ given one
#   sample $(x, y)$, as above.
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

# test the perceptron_step function
def test_perceptron_step():
    x = torch.tensor([1.0, 2.0])
    y = 1.0
    w = torch.tensor([0.5, -0.5])
    b = 0.0
    lr = 0.1

    updated_w, updated_b = perceptron_step(x, y, w, b, lr)
    print("Updated weights:", updated_w)
    print("Updated bias:", updated_b)

# Run the test
test_perceptron_step()
