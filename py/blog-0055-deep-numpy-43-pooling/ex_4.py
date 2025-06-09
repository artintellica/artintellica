import numpy as np
from numpy.typing import NDArray
from typing import Union
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from neural_network import conv2d, softmax, normalize, max_pool


X_full, y_full = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X_full = X_full.astype(float)
batch_size = 2
X_batch = X_full[:batch_size].reshape(batch_size, 28, 28)
labels = y_full[:batch_size]
n_filters = 4
filters = [np.random.randn(3, 3) * 0.1 for _ in range(n_filters)]
feature_maps = np.zeros((batch_size, 26, 26, n_filters))
for i in range(batch_size):
    for f in range(n_filters):
        feature_maps[i, :, :, f] = conv2d(X_batch[i], filters[f], stride=1)
pooled_maps = np.zeros((batch_size, 13, 13, n_filters))
for i in range(batch_size):
    for f in range(n_filters):
        pooled_maps[i, :, :, f] = max_pool(feature_maps[i, :, :, f], size=2, stride=2)
n_flattened = 13 * 13 * n_filters
flattened = pooled_maps.reshape(batch_size, n_flattened)
W_dense = np.random.randn(n_flattened, 10) * 0.01
b_dense = np.zeros((1, 10))
logits = flattened @ W_dense + b_dense
probs = softmax(logits)
print("Input Batch Shape:", X_batch.shape)
print("Feature Maps Shape (after conv):", feature_maps.shape)
print("Pooled Maps Shape (after pooling):", pooled_maps.shape)
print("Flattened Shape:", flattened.shape)
print("Output Probabilities Shape:", probs.shape)
