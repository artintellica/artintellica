import torch
from torch import Tensor
import matplotlib.pyplot as plt


def generate_sine_data(n_samples: int = 100) -> tuple[Tensor, Tensor]:
    # Inputs: shape (N, 1)
    x = torch.linspace(-2 * torch.pi, 2 * torch.pi, n_samples).unsqueeze(1)
    # Outputs: shape (N, 1)
    y = torch.sin(x)
    return x, y


X, y_true = generate_sine_data(300)

D_in = 1         # Input size
D_hidden = 10    # Hidden layer width
D_out = 1        # Output size

# Manual parameter initialization with requires_grad=True
W1 = torch.randn(D_in, D_hidden, requires_grad=True)
b1 = torch.zeros(D_hidden, requires_grad=True)
W2 = torch.randn(D_hidden, D_out, requires_grad=True)
b2 = torch.zeros(D_out, requires_grad=True)
