import numpy as np
from numpy.typing import NDArray
from typing import Union

vec1: NDArray[np.floating] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
vec2: NDArray[np.floating] = np.array([4.0, 5.0], dtype=np.float64)
scalar: Union[float, int] = 2.0

vec1_added: NDArray[np.floating] = vec1 + scalar
vec2_added: NDArray[np.floating] = vec2 + scalar
# vec3: NDArray[np.floating] = vec1 + vec2  # This will raise an error due to shape mismatch
print("Vector vec1 (3 elements):\n", vec1)
print("Vector vec2 (2 elements):\n", vec2)
print("Scalar value to add:", scalar)
print("vec1 + scalar:\n", vec1_added)
print("vec2 + scalar:\n", vec2_added)
# try:
#     print("vec1 + vec2 (should raise an error):\n", vec3)
# except ValueError as e:
#     print("Error when adding vec1 and vec2:", e)
