import torch


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


# Example input
X_example: torch.Tensor = torch.randn(5, 2)
logits_out: torch.Tensor = forward(X_example)
print("Logits (first 5):\n", logits_out)
