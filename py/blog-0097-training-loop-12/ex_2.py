import torch
import matplotlib.pyplot as plt
from typing import List

# Make reproducible
torch.manual_seed(42)

true_w = 2.0  # True slope
true_b = -1.0  # True intercept

N = 100  # Number of data points

# Generate data with two features
torch.manual_seed(42)
true_w = torch.tensor([1.5, -3.0])  # True weights
true_b = 0.5
N = 100

X: torch.Tensor = torch.rand(N, 2) * 2 - 1  # random values in [-1, 1]
y: torch.Tensor = X @ true_w + true_b + 0.1 * torch.randn(N)

# Initialize weights
w: torch.Tensor = torch.randn(2, requires_grad=False)
b: torch.Tensor = torch.randn(1, requires_grad=False)

learning_rate = 0.1
for epoch in range(50):
    y_pred = X @ w + b
    loss = ((y_pred - y) ** 2).mean()

    grad_w = (2 / N) * (X * (y_pred - y).unsqueeze(1)).sum(dim=0)
    grad_b = (2 / N) * (y_pred - y).sum()

    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

print(f"Learned w: {w}")
print(f"True w:    {true_w}")
