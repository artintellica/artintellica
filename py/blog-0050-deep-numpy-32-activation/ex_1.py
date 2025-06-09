import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import relu, sigmoid


Z = np.array([[-1.5, 2.0, 0.0], [-0.5, -2.0, 1.5], [3.0, -1.0, 0.5], [-3.0, 4.0, -0.2]])
A_relu = relu(Z)
print("Input Z (4x3):\n", Z)
print("Output after ReLU (4x3):\n", A_relu)
