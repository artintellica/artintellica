import numpy as np
from numpy.typing import NDArray
from typing import Union

# Compute dot product of two vectors
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
dot_product = np.dot(u, v)
print("Vector u:", u)
print("Vector v:", v)
print("Dot product u · v:", dot_product)
