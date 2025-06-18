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


# 1. Poor initialization (Normal with large std)
print("Normal initialization (std=1):")
plot_activations_hist(
    layer_cls=nn.Linear,
    init_fn=lambda w: nn.init.normal_(w, std=1.0),
    input_shape=(128, 128),
    n_layers=4,
)

# 2. Xavier (Glorot) initialization
print("Xavier/Glorot initialization:")
plot_activations_hist(
    layer_cls=nn.Linear,
    init_fn=lambda w: nn.init.xavier_uniform_(w),
    input_shape=(128, 128),
    n_layers=4,
)

# 3. Kaiming (He) initialization for ReLU
print("Kaiming/He initialization (ReLU):")
plot_activations_hist(
    layer_cls=nn.Linear,
    init_fn=lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"),
    input_shape=(128, 128),
    n_layers=4,
)
