import torch
import matplotlib.pyplot as plt

# Make the output reproducible
torch.manual_seed(42)

N = 100  # Number of data points
X = torch.linspace(0, 1, N).unsqueeze(1)  # Shape: (N, 1)
true_w = torch.tensor([2.0])
true_b = torch.tensor([0.5])
y = X @ true_w + true_b + 0.1 * torch.randn(N, 1)  # Add some noise
