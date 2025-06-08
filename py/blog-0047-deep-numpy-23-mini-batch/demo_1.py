import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable, Tuple, List, cast
# Load MNIST data (subset for digits 0 and 1)
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_network import gradient_descent, binary_cross_entropy, sigmoid, normalize

# Load MNIST dataset (this may take a moment)
X_full: NDArray[np.floating]
y_full: NDArray[np.integer]
X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y_full = y_full.astype(int)
X_full = X_full.astype(float)

# Filter for digits 0 and 1 (binary classification)
mask = (y_full == 0) | (y_full == 1)
X = X_full[mask][:1000]  # Limit to 1000 samples for faster training on CPU
y = y_full[mask][:1000]
y = y.reshape(-1, 1)  # Shape (n_samples, 1)
X = normalize(X)  # Normalize pixel values using our function

# Split into train and test sets
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = cast(NDArray[np.floating], X_train)
X_test = cast(NDArray[np.floating], X_test)
y_train = cast(NDArray[np.floating], y_train)
y_test = cast(NDArray[np.floating], y_test)

# Initialize parameters
n_features = X_train.shape[1]
W_init = np.zeros((n_features, 1))  # Initial weights
b_init = np.zeros((1, 1))  # Initial bias
lr = 0.1  # Learning rate
num_epochs = 10  # Number of epochs
batch_size = 32  # Mini-batch size

# Train logistic regression with sigmoid activation and BCE loss
W_final, b_final, losses = gradient_descent(
    X_train, y_train, W_init, b_init, lr, num_epochs, batch_size,
    loss_fn=binary_cross_entropy, activation_fn=sigmoid
)

# Evaluate on test set
y_pred_test = sigmoid(X_test @ W_final + b_final)
test_loss = binary_cross_entropy(y_pred_test, y_test)
accuracy = np.mean((y_pred_test > 0.5) == y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", accuracy)

