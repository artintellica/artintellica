import numpy as np

X = np.array([[1, 2, 3], [4, 5, 6]])
second_row = X[1, :]
first_two_cols = X[:, 0:2]
print("Second row:", second_row)
print("First two columns:\n", first_two_cols)
