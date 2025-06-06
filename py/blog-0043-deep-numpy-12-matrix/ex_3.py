import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import normalize, matrix_multiply

u: NDArray[np.floating] = np.array([1, 2, 3, 4], dtype=np.float64)
v: NDArray[np.floating] = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
dot_product: np.floating = np.dot(u, v)
print("Vector u:", u)
print("Vector v:", v)
print("Dot product u · v:", dot_product)

u: NDArray[np.floating] = np.array([1, 1, 1, 1], dtype=np.float64)
v: NDArray[np.floating] = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64)
dot_product: np.floating = np.dot(u, v)
print("Vector u:", u)
print("Vector v:", v)
print("Dot product u · v:", dot_product)
