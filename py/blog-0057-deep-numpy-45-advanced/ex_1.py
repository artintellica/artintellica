import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, List, cast
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_network import (
    normalize,
    relu,
    softmax,
    cross_entropy,
    forward_mlp_3layer,
    backward_mlp_3layer,
    l2_regularization,
    dropout,
    accuracy,
    momentum_update,
    accuracy,
)
import matplotlib.pyplot as plt

gradient1 = np.array([[1.0, 2.0], [3.0, 4.0]])
gradient2 = np.array([[0.5, 1.5], [2.5, 3.5]])
velocity = np.zeros((2, 2))
mu = 0.9
lr = 0.1
# First update
velocity, update = momentum_update(velocity, gradient1, mu, lr)
print("First Update - Velocity:\n", velocity)
print("First Update - Parameter Update:\n", update)
# Second update
velocity, update = momentum_update(velocity, gradient2, mu, lr)
print("Second Update - Velocity:\n", velocity)
print("Second Update - Parameter Update:\n", update)
