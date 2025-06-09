import numpy as np
from numpy.typing import NDArray
from typing import Union
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from neural_network import conv2d, softmax, normalize, max_pool


X_full, y_full = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X_full = X_full.astype(float)
image = X_full[0].reshape(28, 28)
label = y_full[0]
filter_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
conv_map = conv2d(image, filter_kernel, stride=1)
pooled_map = max_pool(conv_map, size=2, stride=2)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(image, cmap="gray")
ax1.set_title(f"Input Image (Digit: {label})")
ax1.axis("off")
ax2.imshow(conv_map, cmap="gray")
ax2.set_title("Convolved Feature Map")
ax2.axis("off")
ax3.imshow(pooled_map, cmap="gray")
ax3.set_title("Pooled Feature Map")
ax3.axis("off")
plt.tight_layout()
plt.show()
print("Input Image Shape:", image.shape)
print("Convolved Map Shape:", conv_map.shape)
print("Pooled Map Shape:", pooled_map.shape)
