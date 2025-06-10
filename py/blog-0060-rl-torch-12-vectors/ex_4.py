import torch

# Create a PyTorch vector containing $100$ linearly spaced points between $0$ and $4\pi$.
t: torch.Tensor = torch.linspace(0, 4 * torch.pi, 100)

