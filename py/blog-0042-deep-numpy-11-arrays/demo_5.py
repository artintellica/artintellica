import numpy as np

# Create an array from a list
X = np.array([[1, 2, 3], [4, 5, 6]])

# Access a single element
element = X[0, 1]  # Row 0, Column 1
print("Element at (0,1):", element)

# Slice rows or columns
first_row = X[0, :]  # All columns of row 0
first_column = X[:, 0]  # All rows of column 0
print("First row:", first_row)
print("First column:", first_column)

# Select a submatrix
submatrix = X[0:2, 1:3]  # Rows 0-1, Columns 1-2
print("Submatrix:\n", submatrix)
