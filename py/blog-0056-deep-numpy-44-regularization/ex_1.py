import numpy as np
from numpy.typing import NDArray
from typing import Union, List, cast
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
)
import matplotlib.pyplot as plt

W1 = np.array([[1.0, 2.0], [3.0, 4.0]])
W2 = np.array([[0.5, 1.5], [2.5, 3.5]])
lambda_ = 0.01
l2_penalty, l2_grads = l2_regularization([W1, W2], lambda_)
print("L2 Penalty:", l2_penalty)
print("Gradient for W1:\n", l2_grads[0])
print("Gradient for W2:\n", l2_grads[1])
