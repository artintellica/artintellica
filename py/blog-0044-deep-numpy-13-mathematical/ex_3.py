import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import normalize, matrix_multiply, sigmoid

Z: NDArray[np.floating] = np.array([[-1.5, 2.0], [0.0, -0.5], [3.0, -2.0]], dtype=np.float64)
