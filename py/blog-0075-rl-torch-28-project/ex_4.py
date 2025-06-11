import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


# def make_moons(n_samples=200, noise=0.1):
#     # Generate two interleaving half circles ("moons"), similar to sklearn.datasets.make_moons
#     n = n_samples // 2
#     theta = np.pi * np.random.rand(n)
#     x0 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
#     x1 = np.stack([1 - np.cos(theta), 1 - np.sin(theta)], axis=1) + np.array(
#         [0.6, -0.4]
#     )
#     X = np.vstack([x0, x1])
#     X += noise * np.random.randn(*X.shape)
#     y = np.hstack([np.zeros(n), np.ones(n)])
#     return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# torch.manual_seed(42)
# np.random.seed(42)
# X, y = make_moons(200, noise=0.2)

# model = nn.Sequential(
#     nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 2)  # 2 outputs = logits for 2 classes
# )
# optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
# loss_fn = nn.CrossEntropyLoss()

# # Training loop
# losses = []
# for epoch in range(200):
#     logits = model(X)
#     loss = loss_fn(logits, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     losses.append(loss.item())
#     if epoch % 40 == 0 or epoch == 199:
#         print(f"Epoch {epoch}: loss = {loss.item():.3f}")

# with torch.no_grad():
#     # Create a grid of points covering the data
#     xx, yy = np.meshgrid(
#         np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 200),
#         np.linspace(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, 200),
#     )
#     grid_points = np.c_[xx.ravel(), yy.ravel()]
#     grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
#     logits_grid = model(grid_tensor)
#     probas = torch.softmax(logits_grid, dim=1)
#     preds = probas.argmax(dim=1).cpu().numpy().reshape(xx.shape)

# with torch.no_grad():
#     logits = model(X)
#     predictions = logits.argmax(dim=1)
#     accuracy = (predictions == y).float().mean().item()
#     misclassified = (predictions != y)

# print(f"Accuracy: {accuracy*100:.2f}%")

# plt.scatter(X[y==0,0], X[y==0,1], c='r', label='Class 0', alpha=0.7)
# plt.scatter(X[y==1,0], X[y==1,1], c='b', label='Class 1', alpha=0.7)
# plt.scatter(X[misclassified,0], X[misclassified,1],
#             facecolors='none', edgecolors='k', linewidths=2, s=90, label='Misclassified')
# plt.xlabel("x1"); plt.ylabel("x2")
# plt.legend(); plt.title("Classification Results (Misclassifications circled)")
# plt.show()

# ### **Exercise 1:** Choose or Generate a 2D Dataset for Classification

# - Create or choose a 2D dataset with clear class structure (blobs, moons,
#   circles, or your own idea).
# - Visualize the data, coloring by class.


# for this exercise, we will pick three shapes. the idea is to make three gaussian blobs, and then stretch and rotate them. we will make all 3 blobs first of a similar size. each will be placed with a different center, but slightly overlapping, constituting a triangle. each will be squished and rotated differently, still slightly overlapping, but not too much. let's start with the data first, and then worry about the machine learning model later.
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

# now, we need to find the machine learning model for this data. we will use a simple neural network with one hidden layer, and train it using cross entropy loss. we will use the same training loop as in the previous exercises, but with a different dataset.
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

# # now we need to visualize the decision boundaries learned by the model.
# with torch.no_grad():
#     x1g, x2g = torch.meshgrid(
#         torch.linspace(-6, 6, 200), torch.linspace(-6, 6, 200), indexing="ij"
#     )
#     Xg = torch.stack([x1g.reshape(-1), x2g.reshape(-1)], dim=1)  # (n_grid, 2)
#     logits_grid = model(Xg)
#     y_pred_grid = logits_grid.argmax(dim=1).reshape(200, 200)
# plt.contourf(
#     x1g,
#     x2g,
#     y_pred_grid.numpy(),
#     levels=[-0.5, 0.5, 1.5, 2.5],
#     colors=["b", "r", "g"],
#     alpha=0.15,
# )
# for i in range(3):
#     plt.scatter(
#         blobs_X[blobs_y == i, 0],
#         blobs_X[blobs_y == i, 1],
#         color=["b", "r", "g"][i],
#         alpha=0.6,
#         label=f"Class {i}",
#     )
# plt.title("Learned Class Boundaries (Linear)")
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.legend()
# plt.show()

# now we should plot the incorrectly classified points
with torch.no_grad():
    logits = model(blobs_X)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == blobs_y).float().mean().item()
    misclassified = predictions != blobs_y
print(f"Accuracy: {accuracy*100:.2f}%")
plt.scatter(
    blobs_X[blobs_y == 0, 0],
    blobs_X[blobs_y == 0, 1],
    c="r",
    label="Class 0",
    alpha=0.7,
)
plt.scatter(
    blobs_X[blobs_y == 1, 0],
    blobs_X[blobs_y == 1, 1],
    c="g",
    label="Class 1",
    alpha=0.7,
)
plt.scatter(
    blobs_X[blobs_y == 2, 0],
    blobs_X[blobs_y == 2, 1],
    c="b",
    label="Class 2",
    alpha=0.7,
)
plt.scatter(
    blobs_X[misclassified, 0],
    blobs_X[misclassified, 1],
    facecolors="none",
    edgecolors="k",
    linewidths=2,
    s=90,
    label="Misclassified",
)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Classification Results (Misclassifications circled)")
plt.show()
