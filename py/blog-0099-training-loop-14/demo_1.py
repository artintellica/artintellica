import torch
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(42)

# Generate data
def create_data(n_samples: int = 30) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-3, 3, n_samples)
    y = 2 * x + 1 + torch.randn(n_samples) * 0.5
    return x, y

x, y = create_data()

