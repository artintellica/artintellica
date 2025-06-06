import numpy as np

X = np.array([[1, 2, 3], [4, 5, 6]])
row_vec = np.array([10, 20, 30])
added = X + row_vec
print("X:\n", X)
print("Row vector:\n", row_vec)
print("X + row vector:\n", added)
