import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

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

# Create a TensorDataset and DataLoader for mini-batch processing
dataset = TensorDataset(X, y)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Define the same SimpleNN class as in Example 1
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
model_mini_batch = SimpleNN()
model_full_batch = SimpleNN()  # Separate model for full-batch comparison
print("Model architecture (Mini-Batch):")
print(model_mini_batch)

# Define loss function and optimizers for both models
criterion = nn.MSELoss()
optimizer_mini = torch.optim.SGD(model_mini_batch.parameters(), lr=0.01)
optimizer_full = torch.optim.SGD(model_full_batch.parameters(), lr=0.01)

# Training loop for mini-batch
n_epochs = 100
losses_mini_batch = []
for epoch in range(n_epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        # Forward pass
        y_pred = model_mini_batch(batch_X)
        loss = criterion(y_pred, batch_y)
        epoch_loss += loss.item()

        # Backward pass and optimization
        optimizer_mini.zero_grad()
        loss.backward()
        optimizer_mini.step()

    # Average loss over batches for the epoch
    avg_epoch_loss = epoch_loss / len(dataloader)
    losses_mini_batch.append(avg_epoch_loss)
    if (epoch + 1) % 20 == 0:
        print(
            f"Mini-Batch - Epoch [{epoch+1}/{n_epochs}], Avg Loss: {avg_epoch_loss:.4f}"
        )

# Training loop for full-batch (as in Example 1)
losses_full_batch = []
for epoch in range(n_epochs):
    # Forward pass
    y_pred = model_full_batch(X)
    loss = criterion(y_pred, y)
    losses_full_batch.append(loss.item())

    # Backward pass and optimization
    optimizer_full.zero_grad()
    loss.backward()
    optimizer_full.step()

    if (epoch + 1) % 20 == 0:
        print(f"Full-Batch - Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# Plot loss over epochs for both methods
plt.figure(figsize=(10, 6))
plt.plot(range(n_epochs), losses_mini_batch, label="Mini-Batch Loss (MSE)")
plt.plot(range(n_epochs), losses_full_batch, label="Full-Batch Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs: Mini-Batch vs Full-Batch")
plt.legend()
plt.grid(True)
plt.show()
