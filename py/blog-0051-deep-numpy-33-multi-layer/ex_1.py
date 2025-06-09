import numpy as np
from numpy.typing import NDArray
from typing import Union
from neural_network import relu, softmax, forward_mlp, cross_entropy


X = np.random.randn(2, 3)
W1 = np.random.randn(3, 4) * 0.01
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 2) * 0.01
b2 = np.zeros((1, 2))
A1, A2 = forward_mlp(X, W1, b1, W2, b2)
sums = np.sum(A2, axis=1)
print("Hidden Layer Output A1 shape:", A1.shape)
print("Hidden Layer Output A1:\n", A1)
print("Output Layer Output A2 shape:", A2.shape)
print("Output Layer Output A2:\n", A2)
print("Sum of probabilities per sample (should be ~1):\n", sums)
