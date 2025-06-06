import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import normalize, matrix_multiply, sigmoid

Z: NDArray[np.floating] = np.array([[1, -1], [2, -2]], dtype=np.float64)
