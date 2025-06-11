import torch
import torch.nn.functional as F


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

# Zero gradients (if pre-existing)
for param in [W1, b1, W2, b2]:
    if param.grad is not None:
        param.grad.zero_()

# Forward
logits_batch = forward(X_data)
loss = cross_entropy_loss(logits_batch, y_data)
# Backward: PyTorch computes all gradients automatically
loss.backward()

print("dL/dW1 (shape):", W1.grad.shape if W1.grad is not None else "None")
print("dL/dW2 (shape):", W2.grad.shape if W2.grad is not None else "None")
