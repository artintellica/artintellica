import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)


def softmax(logits: torch.Tensor) -> torch.Tensor:
    logits = logits - logits.max(dim=1, keepdim=True).values  # stability
    exp_logits = torch.exp(logits)
    return exp_logits / exp_logits.sum(dim=1, keepdim=True)


# Network sizes
input_dim: int = 2
hidden_dim: int = 8
output_dim: int = 2  # binary classification

# Weights and biases
W1: torch.Tensor = torch.randn(input_dim, hidden_dim, requires_grad=True)  # (2, 8)
b1: torch.Tensor = torch.zeros(hidden_dim, requires_grad=True)  # (8,)
W2: torch.Tensor = torch.randn(hidden_dim, output_dim, requires_grad=True)  # (8, 2)
b2: torch.Tensor = torch.zeros(output_dim, requires_grad=True)  # (2,)


# Forward, example for a batch X of shape (N, 2)
def forward(X: torch.Tensor) -> torch.Tensor:
    z1: torch.Tensor = X @ W1 + b1  # (N, 8)
    h: torch.Tensor = relu(z1)  # (N, 8)
    logits: torch.Tensor = h @ W2 + b2  # (N, 2)
    return logits


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


# Synthetic dataset
N: int = 80
X_data: torch.Tensor = torch.randn(N, 2)
# True decision boundary: if x0 + x1 > 0, class 1 else 0
y_data: torch.Tensor = ((X_data[:, 0] + X_data[:, 1]) > 0).long()

lr: float = 0.07
epochs: int = 120
losses: list[float] = []

# Reinitialize weights/biases for training run
W1 = torch.randn(input_dim, hidden_dim, requires_grad=True)
b1 = torch.zeros(hidden_dim, requires_grad=True)
W2 = torch.randn(hidden_dim, output_dim, requires_grad=True)
b2 = torch.zeros(output_dim, requires_grad=True)

for epoch in range(epochs):
    # Forward
    logits = forward(X_data)
    loss = cross_entropy_loss(logits, y_data)
    # Backward
    loss.backward()
    # Gradient descent
    with torch.no_grad():
        W1 -= lr * W1.grad if W1.grad is not None else 0
        b1 -= lr * b1.grad if b1.grad is not None else 0
        W2 -= lr * W2.grad if W2.grad is not None else 0
        b2 -= lr * b2.grad if b2.grad is not None else 0
    for param in [W1, b1, W2, b2]:
        param.grad.zero_()
    losses.append(loss.item())
    if epoch % 30 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}: Loss={loss.item():.3f}")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Feedforward NN Training Loss")
plt.grid(True)
plt.show()
