import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Type


def plot_activations_hist(
    layer_cls: Type[nn.Module],
    init_fn,
    input_shape: tuple,
    n_layers: int,
    N: int = 1000,
) -> None:
    """Visualize distribution of activations across layers after different initializations."""
    torch.manual_seed(42)
    layers = []
    for _ in range(n_layers):
        layer = layer_cls(*input_shape)
        init_fn(layer.weight)
        nn.init.zeros_(layer.bias)  # type: ignore
        layers.append(layer)
    x = torch.randn(N, input_shape[0])
    activations = [x]
    for layer in layers:
        x = torch.relu(layer(x))
        activations.append(x)
    plt.figure(figsize=(12, 4))
    for i, a in enumerate(activations[1:], 1):
        plt.subplot(1, n_layers, i)
        plt.hist(a.detach().numpy().ravel(), bins=40, color="skyblue")
        plt.title(f"Layer {i}")
        plt.xlabel("Activation")
        plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


import torch.nn as nn


class MyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc2.weight)
        # Bias defaults to zeros

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class ManualInitNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ManualInitNet()
print(model)
