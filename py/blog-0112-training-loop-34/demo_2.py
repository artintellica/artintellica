import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Fix random seed for reproducibility
torch.manual_seed(0)


class DeepSigmoidNet(nn.Module):
    def __init__(self, depth: int, hidden_size: int = 32):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(depth)]
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


# Build a deep network
depth = 20
net = DeepSigmoidNet(depth=depth)
x = torch.randn(16, 32)
target = torch.randn(16, 32)

# Forward pass
output = net(x)
loss = (output - target).pow(2).mean()

# Backward pass
loss.backward()

# Record gradient norms for each layer
grad_norms = []
for i, layer in enumerate(net.layers):
    norm = layer.weight.grad.norm().item()  # type: ignore
    grad_norms.append(norm)

plt.plot(range(1, depth + 1), grad_norms, marker="o")
plt.xlabel("Layer")
plt.ylabel("Weight Gradient Norm")
plt.title("Vanishing Gradients in Deep Sigmoid Network")
plt.show()


class DeepBatchNormSigmoidNet(nn.Module):
    def __init__(self, depth: int, hidden_size: int = 32):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size)
                )
                for _ in range(depth)
            ]
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


# Try again
net_bn = DeepBatchNormSigmoidNet(depth=depth)
x = torch.randn(16, 32)
target = torch.randn(16, 32)
output = net_bn(x)
loss = (output - target).pow(2).mean()
loss.backward()

# New gradient norms
grad_norms_bn = [layer[0].weight.grad.norm().item() for layer in net_bn.layers] # type: ignore

plt.plot(
    range(1, depth + 1), grad_norms_bn, marker="s", label="BatchNorm", color="orange"
)
plt.plot(
    range(1, depth + 1),
    grad_norms,
    marker="o",
    linestyle="--",
    label="Vanilla",
    alpha=0.6,
)
plt.xlabel("Layer")
plt.ylabel("Weight Gradient Norm")
plt.title("Gradient Norms: BatchNorm vs No BatchNorm")
plt.legend()
plt.show()
