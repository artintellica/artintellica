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

X = np.array([[0.5], [1.5], [1.0], [2.0], [3.0], [2.5]])
y = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
W_init = np.array([[0.0]])
b_init = np.array([[0.0]])
lr = 0.01
num_epochs = 10
batch_size = 2
W_final, b_final, losses = gradient_descent(
    X, y, W_init, b_init, lr, num_epochs, batch_size, binary_cross_entropy, sigmoid
)
print("Initial weight W:", W_init)
print("Initial bias b:", b_init)
print("Final weight W:", W_final)
print("Final bias b:", b_final)
print("Loss history:", losses)
