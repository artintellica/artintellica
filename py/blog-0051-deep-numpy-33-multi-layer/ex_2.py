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

# Assuming A2 from Exercise 1
X = np.random.randn(2, 3)
W1 = np.random.randn(3, 4) * 0.01
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 2) * 0.01
b2 = np.zeros((1, 2))
_, A2 = forward_mlp(X, W1, b1, W2, b2)
y = np.array([[1, 0], [0, 1]])
loss = cross_entropy(A2, y)
print("Output Probabilities A2:\n", A2)
print("True Labels y:\n", y)
print("Cross-Entropy Loss:", loss)
