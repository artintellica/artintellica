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


