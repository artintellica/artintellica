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
horizontal_filter = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
vertical_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
horizontal_map = conv2d(image, horizontal_filter, stride=1)
vertical_map = conv2d(image, vertical_filter, stride=1)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(image, cmap="gray")
ax1.set_title(f"Input Image (Digit: {label})")
ax1.axis("off")
ax2.imshow(horizontal_map, cmap="gray")
ax2.set_title("Feature Map (Horizontal Edges)")
ax2.axis("off")
ax3.imshow(vertical_map, cmap="gray")
ax3.set_title("Feature Map (Vertical Edges)")
ax3.axis("off")
plt.tight_layout()
plt.show()
print("Input Image Shape:", image.shape)
print("Horizontal Feature Map Shape:", horizontal_map.shape)
print("Vertical Feature Map Shape:", vertical_map.shape)
