import numpy as np
from numpy.typing import NDArray
from typing import Union
from scipy import signal
from neural_network import conv2d

# Your code here
image = np.array([[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]])
filter_kernel = np.array([[0.25, 0.25], [0.25, 0.25]])
feature_map = conv2d(image, filter_kernel, stride=2)
print("Input Image (4x4):\n", image)
print("Output Feature Map (stride=2):\n", feature_map)
print("Output Shape:", feature_map.shape)
