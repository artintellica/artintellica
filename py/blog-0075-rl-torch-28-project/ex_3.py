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

# now let's visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(
    blobs_X[blobs_y == 0, 0],
    blobs_X[blobs_y == 0, 1],
    c="r",
    label="Class 0",
    alpha=0.6,
)
plt.scatter(
    blobs_X[blobs_y == 1, 0],
    blobs_X[blobs_y == 1, 1],
    c="g",
    label="Class 1",
    alpha=0.6,
)
plt.scatter(
    blobs_X[blobs_y == 2, 0],
    blobs_X[blobs_y == 2, 1],
    c="b",
    label="Class 2",
    alpha=0.6,
)

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

# now we need to visualize the decision boundaries learned by the model.
with torch.no_grad():
    x1g, x2g = torch.meshgrid(
        torch.linspace(-6, 6, 200), torch.linspace(-6, 6, 200), indexing="ij"
    )
    Xg = torch.stack([x1g.reshape(-1), x2g.reshape(-1)], dim=1)  # (n_grid, 2)
    logits_grid = model(Xg)
    y_pred_grid = logits_grid.argmax(dim=1).reshape(200, 200)
plt.contourf(
    x1g,
    x2g,
    y_pred_grid.numpy(),
    levels=[-0.5, 0.5, 1.5, 2.5],
    colors=["b", "r", "g"],
    alpha=0.15,
)
for i in range(3):
    plt.scatter(
        blobs_X[blobs_y == i, 0],
        blobs_X[blobs_y == i, 1],
        color=["b", "r", "g"][i],
        alpha=0.6,
        label=f"Class {i}",
    )
plt.title("Learned Class Boundaries (Linear)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()
