import numpy as np
from numpy.typing import NDArray
from typing import Union
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from neural_network import conv2d, softmax, normalize, max_pool


X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
pooled = max_pool(X, size=2, stride=2)
print("Input Feature Map (4x4):\n", X)
print("Output after Max Pooling (2x2):\n", pooled)
print("Output Shape:", pooled.shape)
