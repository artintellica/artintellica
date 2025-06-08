import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable
from neural_network import gradient_descent, mse_loss

# Example: Synthetic data (y = 2x + 1)
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])  # Input (5 samples, 1 feature)
y = np.array([[3.0], [5.0], [7.0], [9.0], [11.0]])  # True values (y = 2x + 1)
W_init = np.array([[0.0]])  # Initial weight (start far from true value 2.0)
b_init = np.array([[0.0]])  # Initial bias (start far from true value 1.0)
lr = 0.1  # Learning rate
num_iterations = 100  # Number of iterations


# Run gradient descent
W_final, b_final, losses = gradient_descent(
    X, y, W_init, b_init, lr, num_iterations, mse_loss
)

print("Initial weight W:", W_init)
print("Initial bias b:", b_init)
print("Final weight W after training:", W_final)
print("Final bias b after training:", b_final)
print("Final loss:", losses[-1])
print("First few losses:", losses[:5])
