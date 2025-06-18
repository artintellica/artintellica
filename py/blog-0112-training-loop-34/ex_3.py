import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):
    def __init__(self, core_layer: nn.Module):
        super().__init__()
        self.core_layer = core_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core_layer(x) + x  # Residual connection


# Example usage
block = ResidualBlock(nn.Linear(32, 32))
x = torch.randn(8, 32)
output = block(x)
print("Output shape:", output.shape)
