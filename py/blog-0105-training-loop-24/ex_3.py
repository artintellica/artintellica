import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def swish(z: torch.Tensor) -> torch.Tensor:
    return z * torch.sigmoid(z)


# Visualize Swish
z = torch.linspace(-5, 5, steps=200)
plt.plot(z.numpy(), swish(z).numpy(), label="Swish")
plt.legend()
plt.grid(True)
plt.title("Swish Activation Function")
plt.show()
