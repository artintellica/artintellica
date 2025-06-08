import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable
from neural_network import gradient_descent, mse_loss

X: NDArray[np.floating] = np.array([[1.0], [2.0], [3.0]])
y: NDArray[np.floating] = np.array([[2.0], [4.0], [6.0]])  # True values (y = 2x)
W_init = np.array([[0.0]])
b_init = np.array([[0.0]])  # Initial bias
lr: float = 0.01  # Learning rate
num_iterations: int = 50  # Number of iterations
W_final, b_final, losses = gradient_descent(
    X, y, W_init, b_init, lr, num_iterations, mse_loss
)
print("Initial weight W:", W_init)
print("Initial bias b:", b_init)
print("Final weight W:", W_final)
print("Final bias b:", b_final)
print("Final loss:", losses[-1])
# print("Loss history:", losses)
