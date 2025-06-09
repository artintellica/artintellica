import numpy as np
from numpy.typing import NDArray
from typing import Union
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from neural_network import conv2d, softmax, normalize


def max_pool(
    X: NDArray[np.floating], size: int = 2, stride: int = 2
) -> NDArray[np.floating]:
    """
    Perform 2D max pooling on an input feature map.
    Args:
        X: Input feature map, shape (height, width) or (height, width, channels)
        size: Size of the pooling window (default: 2 for 2x2 pooling)
        stride: Stride of the pooling operation (default: 2)
    Returns:
        Output after max pooling, shape depends on input, size, and stride
    """
    if len(X.shape) == 2:
        height, width = X.shape
        channels = 1
        X = X.reshape(height, width, 1)
    else:
        height, width, channels = X.shape

    out_height = (height - size) // stride + 1
    out_width = (width - size) // stride + 1
    output = np.zeros((out_height, out_width, channels))

    for i in range(out_height):
        for j in range(out_width):
            x_start = i * stride
            x_end = x_start + size
            y_start = j * stride
            y_end = y_start + size
            region = X[x_start:x_end, y_start:y_end, :]
            output[i, j, :] = np.max(region, axis=(0, 1))

    if channels == 1:
        output = output[:, :, 0]
    return output


# Load MNIST data (small batch for simplicity)
print("Loading MNIST data...")
X_full, y_full = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X_full = X_full.astype(float)
y_full = y_full.astype(int)

# Use a small batch of 4 images
batch_size = 4
X_batch = X_full[:batch_size].reshape(batch_size, 28, 28)  # Shape (4, 28, 28)
labels = y_full[:batch_size]

# Define 8 filters of size 3x3 (random for demo)
n_filters = 8
filters = [np.random.randn(3, 3) * 0.1 for _ in range(n_filters)]

# Step 1: Convolutional Layer
feature_maps = np.zeros(
    (batch_size, 26, 26, n_filters)
)  # Output shape after 3x3 valid conv
for i in range(batch_size):
    for f in range(n_filters):
        feature_maps[i, :, :, f] = conv2d(X_batch[i], filters[f], stride=1)

# Step 2: Max Pooling Layer (2x2, stride=2)
pooled_maps = np.zeros(
    (batch_size, 13, 13, n_filters)
)  # Output shape after 2x2 pooling
for i in range(batch_size):
    for f in range(n_filters):
        pooled_maps[i, :, :, f] = max_pool(feature_maps[i, :, :, f], size=2, stride=2)

# Step 3: Flatten and Dense Layer (for demo, random weights)
n_flattened = 13 * 13 * n_filters  # 13x13x8 = 2197
flattened = pooled_maps.reshape(batch_size, n_flattened)
W_dense = np.random.randn(n_flattened, 10) * 0.01  # 10 classes for MNIST
b_dense = np.zeros((1, 10))
logits = flattened @ W_dense + b_dense
probs = softmax(logits)

# Visualize one input image and one feature map after pooling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(X_batch[0], cmap="gray")
ax1.set_title(f"Input Image (Digit: {labels[0]})")
ax1.axis("off")

ax2.imshow(pooled_maps[0, :, :, 0], cmap="gray")
ax2.set_title("Pooled Feature Map (Filter 1)")
ax2.axis("off")

plt.tight_layout()
plt.show()

print("Input Batch Shape:", X_batch.shape)
print("Feature Maps Shape (after conv):", feature_maps.shape)
print("Pooled Maps Shape (after pooling):", pooled_maps.shape)
print("Flattened Shape:", flattened.shape)
print("Output Probabilities Shape:", probs.shape)
print("Output Probabilities (first sample, first few classes):\n", probs[0, :3])
