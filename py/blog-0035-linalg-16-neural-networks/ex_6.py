import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility (same as Exercise 5)
np.random.seed(53)
torch.manual_seed(53)

# Create the same synthetic 2D dataset for binary classification (100 samples)
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

# Create a DataLoader for mini-batch processing to inspect the last batch
batch_size = 20
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Define the same neural network as in Exercise 5
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

# Training loop with mini-batches
n_epochs = 200
losses = []
last_batch_X = None
last_batch_y = None

for epoch in range(n_epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        # Forward pass
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        epoch_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store the last batch of the last epoch for gradient inspection
        if epoch == n_epochs - 1:
            last_batch_X = batch_X
            last_batch_y = batch_y

    # Average loss over batches for the epoch
    avg_epoch_loss = epoch_loss / len(dataloader)
    losses.append(avg_epoch_loss)
    if (epoch + 1) % 40 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], Avg Loss: {avg_epoch_loss:.4f}")

# Compute gradients for the last batch of the last epoch
# Forward pass on the last batch
model.train()  # Ensure model is in training mode
y_pred_last = model(last_batch_X)
loss_last = criterion(y_pred_last, last_batch_y)

# Backward pass to compute gradients
optimizer.zero_grad()
loss_last.backward()

# Print the gradients of the first layer's weights
print("\nGradients of loss with respect to layer1 weights (layer1.weight.grad):")
print(model.layer1.weight.grad)

# Print the magnitude (L2 norm) of the gradients for analysis
grad_magnitude = torch.norm(model.layer1.weight.grad).item()
print(f"\nL2 Norm (magnitude) of layer1.weight gradients: {grad_magnitude:.6f}")

# Comment on the magnitude of the gradients
print("\nComment on gradient magnitude:")
if grad_magnitude < 0.01:
    print(
        "The gradient magnitude is very small (< 0.01), suggesting the model has likely converged or is close to convergence, as the updates to weights are minimal."
    )
elif grad_magnitude < 0.1:
    print(
        "The gradient magnitude is small (< 0.1), indicating the model is likely approaching convergence, with smaller updates to weights."
    )
else:
    print(
        "The gradient magnitude is relatively large (>= 0.1), suggesting the model may not have fully converged yet, as significant updates to weights are still occurring."
    )

# Plot loss over epochs
plt.figure(figsize=(8, 6))
plt.plot(range(n_epochs), losses, label="Training Loss (BCE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs (Binary Classification)")
plt.legend()
plt.grid(True)
plt.show()
