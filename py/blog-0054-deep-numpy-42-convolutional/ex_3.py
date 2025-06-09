import numpy as np
from numpy.typing import NDArray
from typing import Union
from scipy import signal
from neural_network import conv2d

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

X_full, y_full = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X_full = X_full.astype(float)
image = X_full[0].reshape(28, 28)
label = y_full[0]
filter_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
feature_map = conv2d(image, filter_kernel, stride=1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image, cmap="gray")
ax1.set_title(f"Input Image (Digit: {label})")
ax1.axis("off")
ax2.imshow(feature_map, cmap="gray")
ax2.set_title("Feature Map (Horizontal Edges)")
ax2.axis("off")
plt.tight_layout()
plt.show()
print("Input Image Shape:", image.shape)
print("Feature Map Shape:", feature_map.shape)
