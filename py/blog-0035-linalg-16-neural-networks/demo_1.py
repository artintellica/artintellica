import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic 2D data for regression
n_samples = 200
X = np.random.randn(n_samples, 2) * 2  # 2 features
y = 0.5 * X[:, 0] ** 2 + 1.5 * X[:, 1] + 2.0 + np.random.randn(n_samples) * 0.5
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).reshape(-1, 1)
print("Data shape:", X.shape, y.shape)


# Define a simple neural network with one hidden layer
class SimpleNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=10, output_size=1):
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
print("Model architecture:")
print(model)

# Print parameter shapes
print("\nParameter shapes:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
n_epochs = 200
losses = []
for epoch in range(n_epochs):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)
    losses.append(loss.item())

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# Plot loss over epochs
plt.figure(figsize=(8, 6))
plt.plot(range(n_epochs), losses, label="Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs (Simple Neural Network)")
plt.legend()
plt.grid(True)
plt.show()
