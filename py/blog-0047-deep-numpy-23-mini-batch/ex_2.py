import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable, Tuple, List, cast
from neural_network import (
    gradient_descent,
    binary_cross_entropy,
    sigmoid,
    normalize,
    mse_loss,
)

X: NDArray[np.floating] = np.array(
    [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]
)
y: NDArray[np.floating] = np.array(
    [[2.0], [4.0], [6.0], [8.0], [10.0], [12.0], [14.0], [16.0]]
)
W_init = np.array([[0.0]])
b_init = np.array([[0.0]])
lr = 0.01
num_epochs = 5
batch_size = 4
W_final, b_final, losses = gradient_descent(
    X, y, W_init, b_init, lr, num_epochs, batch_size, mse_loss
)
print("Initial weight W:", W_init)
print("Initial bias b:", b_init)
print("Final weight W:", W_final)
print("Final bias b:", b_final)
print("Loss history:", losses)
