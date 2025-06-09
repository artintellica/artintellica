import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import relu, sigmoid, softmax

Z = np.random.uniform(-5, 5, (5, 4))
A_relu = relu(Z)
zero_count = np.sum(A_relu == 0)
print("Input Z (5x4):\n", Z)
print("Output after ReLU (5x4):\n", A_relu)
print("Number of elements set to 0:", zero_count)
