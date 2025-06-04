import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility (same as Example 1)
torch.manual_seed(42)
np.random.seed(42)

# Generate the same synthetic 2D data for regression as in Example 1
n_samples = 200
X = np.random.randn(n_samples, 2) * 2  # 2 features
y = 0.5 * X[:, 0] ** 2 + 1.5 * X[:, 1] + 2.0 + np.random.randn(n_samples) * 0.5
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).reshape(-1, 1)
print("Data shape:", X.shape, y.shape)


# Define a custom activation function (scaled tanh: 2 * tanh(x))
def scaled_tanh(x):
    return 2 * torch.tanh(x)


# Extend the SimpleNN class to include the custom activation function
class CustomNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=10, output_size=1):
        super(CustomNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.custom_activation = scaled_tanh
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.custom_activation(x)
        x = self.layer2(x)
        return x


# Instantiate the model
model = CustomNN()
print("Model architecture:")
print(model)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
n_epochs = 100
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

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# Plot loss over epochs
plt.figure(figsize=(8, 6))
plt.plot(range(n_epochs), losses, label="Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs (Custom Activation: 2 * tanh)")
plt.legend()
plt.grid(True)
plt.show()
