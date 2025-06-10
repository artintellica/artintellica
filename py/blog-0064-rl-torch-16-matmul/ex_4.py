import torch
import matplotlib.pyplot as plt


# - Intentionally attempt $X @ Y$ where $X$ is $3 \times 2$ and $Y$ is
#   $3 \times 2$ (not allowed).
# - Print the error.
# - Fix the error by transposing $Y$ or $X$ and perform the multiplication
#   successfully.
X: torch.Tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
Y: torch.Tensor = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
try:
    # Attempting to multiply incompatible matrices
    result = X @ Y
except RuntimeError as err:
    print("Shape mismatch error:", err)

Y_T: torch.Tensor = Y.T  # Transpose Y to make it compatible
result_fixed: torch.Tensor = X @ Y_T
print("Fixed result (X @ Y.T):\n", result_fixed)
