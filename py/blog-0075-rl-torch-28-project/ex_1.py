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
plt.title("Three Blobs Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()
plt.show()
