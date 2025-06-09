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

y_pred = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
y_true_one_hot = np.array([[1, 0], [0, 1], [1, 0]])
y_true_indices = np.array([0, 1, 0])
acc_one_hot = accuracy(y_pred, y_true_one_hot)
acc_indices = accuracy(y_pred, y_true_indices)
print("Accuracy with one-hot labels:", acc_one_hot)
print("Accuracy with index labels:", acc_indices)
