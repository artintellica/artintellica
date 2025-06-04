import numpy as np
import torch
import torch.nn as nn

# Set random seed for reproducibility
np.random.seed(52)
torch.manual_seed(52)

# Create a small dataset (10 samples, 2 features) with NumPy
n_samples = 10
X_np = np.random.randn(n_samples, 2) * 1.5  # 2 features
print("Dataset shape (NumPy):", X_np.shape)

# Convert to PyTorch tensor
X = torch.FloatTensor(X_np)
print("Dataset shape (PyTorch):", X.shape)

# Manual implementation of forward pass with matrix operations
# Define weights and biases for one hidden layer (4 units) and output layer (1 unit)
input_size = 2
hidden_size = 4
output_size = 1

# Initialize weights and biases manually (random values for demonstration)
W1 = torch.randn(hidden_size, input_size) * 0.1  # Shape: (4, 2)
b1 = torch.zeros(hidden_size)  # Shape: (4,)
W2 = torch.randn(output_size, hidden_size) * 0.1  # Shape: (1, 4)
b2 = torch.zeros(output_size)  # Shape: (1,)

print("\nManual parameter shapes:")
print("W1 shape:", W1.shape)
print("b1 shape:", b1.shape)
print("W2 shape:", W2.shape)
print("b2 shape:", b2.shape)

# Manual forward pass using matrix multiplications
Z1 = torch.matmul(X, W1.T) + b1  # Shape: (10, 4)
H1 = torch.relu(Z1)  # ReLU activation, Shape: (10, 4)
Z2 = torch.matmul(H1, W2.T) + b2  # Shape: (10, 1)
output_manual = Z2
print("\nManual forward pass output shape:", output_manual.shape)
print("Manual forward pass output (first 3 samples):")
print(output_manual[:3])


# PyTorch nn.Linear implementation for comparison
# Define the same network using nn.Linear
class SimpleNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# Instantiate the model
model = SimpleNN()

# Set the weights and biases to match the manual ones for fair comparison
with torch.no_grad():
    model.layer1.weight.copy_(W1)
    model.layer1.bias.copy_(b1)
    model.layer2.weight.copy_(W2)
    model.layer2.bias.copy_(b2)

# Forward pass using nn.Linear
output_model = model(X)
print("\nPyTorch nn.Linear forward pass output shape:", output_model.shape)
print("PyTorch nn.Linear forward pass output (first 3 samples):")
print(output_model[:3])

# Compare the outputs
difference = torch.abs(output_manual - output_model)
print("\nMax difference between manual and nn.Linear outputs:", difference.max().item())
print(
    "Are outputs nearly equal (within 1e-6)?",
    torch.allclose(output_manual, output_model, atol=1e-6),
)
