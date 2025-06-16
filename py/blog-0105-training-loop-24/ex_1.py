import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_activation_and_derivative() -> None:
    z = torch.linspace(-5, 5, steps=200)
    activations = {
        "ReLU": (F.relu(z), (z > 0).float()),
        "Sigmoid": (torch.sigmoid(z), torch.sigmoid(z) * (1 - torch.sigmoid(z))),
        "Tanh": (torch.tanh(z), 1 - torch.tanh(z) ** 2),
    }
    plt.figure(figsize=(10, 6))
    for idx, (name, (val, grad)) in enumerate(activations.items()):
        plt.subplot(2, 3, idx + 1)
        plt.plot(z.numpy(), val.numpy())
        plt.title(f"{name}")
        plt.subplot(2, 3, idx + 4)
        plt.plot(z.numpy(), grad.numpy())
        plt.title(f"{name} Derivative")
    plt.tight_layout()
    plt.show()


plot_activation_and_derivative()
