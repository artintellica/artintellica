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

A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
p = 0.5
# First run with training=True
A_drop1 = dropout(A, p, training=True)
print("First Dropout (training=True):\n", A_drop1)
# Second run with training=True (different randomness)
A_drop2 = dropout(A, p, training=True)
print("Second Dropout (training=True):\n", A_drop2)
# Run with training=False (no dropout)
A_no_drop = dropout(A, p, training=False)
print("No Dropout (training=False):\n", A_no_drop)
