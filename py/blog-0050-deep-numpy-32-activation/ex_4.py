import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import relu, sigmoid, softmax

Z = np.array([[1000, 1000, 1000], [-1000, -1000, -1000]])
A_softmax = softmax(Z)
sums = np.sum(A_softmax, axis=1)
print("Input Z (2x3, extreme values):\n", Z)
print("Output after Softmax (2x3):\n", A_softmax)
print("Sum of probabilities per sample (should be ~1):\n", sums)
