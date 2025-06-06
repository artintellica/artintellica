import numpy as np
from numpy.typing import NDArray
from typing import Union


# Apply ReLU-like operation using np.maximum
Z = np.array([[-1.0, 0.0, 1.0], [-2.0, 3.0, -0.5]])
print("Input matrix Z (2x3):\n", Z)
A_relu = np.maximum(0, Z)
print("ReLU-like output (max(0, Z)):\n", A_relu)
