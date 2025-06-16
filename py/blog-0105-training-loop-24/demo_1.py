
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def plot_activations() -> None:
    z = torch.linspace(-5, 5, steps=200)

    activations = {
        "ReLU": F.relu(z),
        "Sigmoid": torch.sigmoid(z),
        "Tanh": torch.tanh(z),
    }

    plt.figure(figsize=(8, 5))
    for name, activation in activations.items():
        plt.plot(z.numpy(), activation.numpy(), label=name)
    plt.title("Common Activation Functions")
    plt.xlabel("Input z")
    plt.ylabel("Activation(z)")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_activations()
