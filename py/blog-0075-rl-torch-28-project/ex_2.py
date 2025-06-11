import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


# for this exercise, we will pick three shapes. the idea is to make three
# gaussian blobs, and then stretch and rotate them. we will make all 3 blobs
# first of a similar size. each will be placed with a different center, but
# slightly overlapping, constituting a triangle. each will be squished and
# rotated differently, still slightly overlapping, but not too much. let's
# start with the data first, and then worry about the machine learning model
# later.
def make_three_blobs(n_samples=200, noise=0.1):
    # Generate three blobs in a triangular arrangement
    n = n_samples // 3
    np.random.seed(42)

    # Blob centers
    centers = np.array(
        [
            [0, 0],  # Center of first blob
            [2, 2],  # Center of second blob
            [4, 0],  # Center of third blob
        ]
    )

    # Generate blobs with different spreads and rotations
    X = []
    y = []

    for i in range(3):
        theta = np.pi / 6 * i  # Rotate each blob differently
        x_blob = np.random.randn(n) * 0.5 + centers[i, 0]
        y_blob = np.random.randn(n) * 0.5 + centers[i, 1]

        # Rotate the blob points
        x_rotated = x_blob * np.cos(theta) - y_blob * np.sin(theta)
        y_rotated = x_blob * np.sin(theta) + y_blob * np.cos(theta)

        X.append(np.column_stack((x_rotated, y_rotated)))
        y.extend([i] * n)

    X = np.vstack(X)
    y = np.array(y)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


blobs_X, blobs_y = make_three_blobs(300, noise=0.1)

# now, we need to find the machine learning model for this data. we will use a
# simple neural network with one hidden layer, and train it using cross entropy
# loss. we will use the same training loop as in the previous exercises, but
# with a different dataset.
model = nn.Sequential(
    nn.Linear(2, 10),  # Input layer with 2 features, hidden layer with 10 neurons
    nn.ReLU(),  # Activation function
    nn.Linear(10, 3),  # Output layer with 3 classes (logits)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()
# Training loop
losses = []
for epoch in range(200):
    logits = model(blobs_X)
    loss = loss_fn(logits, blobs_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 40 == 0 or epoch == 199:
        print(f"Epoch {epoch}: loss = {loss.item():.3f}")
# Plot training loss
plt.figure(figsize=(8, 6))
plt.plot(losses, label="Training Loss")
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid()
plt.legend()
plt.show()
