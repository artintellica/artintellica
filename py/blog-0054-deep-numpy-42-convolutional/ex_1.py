import numpy as np
from numpy.typing import NDArray
from typing import Union
from scipy import signal
from neural_network import conv2d


import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load MNIST data (single image for simplicity)
print("Loading MNIST data...")
X_full, y_full = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X_full = X_full.astype(float)
image = X_full[0].reshape(28, 28)  # First image, shape (28, 28)
label = y_full[0]

# Define a simple 3x3 filter (e.g., edge detection)
filter_kernel = np.array(
    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
)  # Sobel-like filter for edges

# Apply convolution with stride=1
feature_map = conv2d(image, filter_kernel, stride=1)

# Visualize input image and output feature map
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image, cmap="gray")
ax1.set_title(f"Input Image (Digit: {label})")
ax1.axis("off")

ax2.imshow(feature_map, cmap="gray")
ax2.set_title("Feature Map (Edge Detection)")
ax2.axis("off")

plt.tight_layout()
plt.show()

print("Input Image Shape:", image.shape)
print("Feature Map Shape:", feature_map.shape)
