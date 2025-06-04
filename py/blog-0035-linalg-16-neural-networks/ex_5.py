import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(53)
torch.manual_seed(53)

# Create a synthetic 2D dataset for binary classification (100 samples)
n_samples = 100
# Generate two classes with some overlap
class0 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array(
    [-1.5, -1.5]
)  # Class 0 centered at (-1.5, -1.5)
class1 = np.random.randn(n_samples // 2, 2) * 0.5 + np.array(
    [1.5, 1.5]
)  # Class 1 centered at (1.5, 1.5)
X = np.vstack([class0, class1])
y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

# Convert to PyTorch tensors
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).reshape(-1, 1)
print("Data shape:", X.shape, y.shape)


# Define a neural network with one hidden layer and sigmoid output
class BinaryClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=5, output_size=1):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x


# Instantiate the model
model = BinaryClassifier()
print("Model architecture:")
print(model)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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

    if (epoch + 1) % 40 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# Plot loss over epochs
plt.figure(figsize=(8, 6))
plt.plot(range(n_epochs), losses, label="Training Loss (BCE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs (Binary Classification)")
plt.legend()
plt.grid(True)
plt.show()
