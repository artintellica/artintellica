import torch
from torch import nn
from typing import Tuple

# Set manual seed for reproducible results
torch.manual_seed(0)

# Define a simple feedforward neural network with 1 hidden layer
class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Dummy data: 2 samples, 3 features each
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y_true = torch.tensor([[10.0], [20.0]])

# Instantiate the network
model = SimpleNN(input_dim=3, hidden_dim=4, output_dim=1)

# Forward Pass
y_pred = model(X)
print('Predictions:', y_pred.detach().numpy())

# Compute loss (Mean Squared Error)
criterion = nn.MSELoss()
loss = criterion(y_pred, y_true)
print('Loss:', loss.item())

# Backward Pass
model.zero_grad()     # Clears old gradients
loss.backward()       # Computes gradients

# Inspect gradients
print('Gradient of layer1 weights:')
print(model.layer1.weight.grad)
print('Gradient of layer2 weights:')
print(model.layer2.weight.grad)

