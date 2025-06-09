import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable, Tuple, List, Dict, cast
from scipy import signal


def normalize(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Normalize the input array X by subtracting the mean and dividing by the standard deviation.
    Parameters:
        X (NDArray[np.floating]): Input array to normalize. Should be a numerical array
            (float or compatible type).
    Returns:
        NDArray[np.floating]: Normalized array with mean approximately 0 and standard
            deviation approximately 1 along each axis.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    normalized_X = X - mean  # Start with mean subtraction
    mask = std != 0  # Create a mask for non-zero std
    # Only apply division where std != 0
    if np.any(mask):
        normalized_X[:, mask] = normalized_X[:, mask] / std[mask]
    return normalized_X


def matrix_multiply(
    X: NDArray[np.floating], W: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Perform matrix multiplication between two arrays.
    Args:
        X: First input array/matrix of shape (m, n) with floating-point values
        W: Second input array/matrix of shape (n, p) with floating-point values
    Returns:
        Result of matrix multiplication, shape (m, p) with floating-point values
    """
    return np.matmul(X, W)


def sigmoid(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the sigmoid activation function element-wise.
    Args:
        Z: Input array of any shape with floating-point values
    Returns:
        Array of the same shape with sigmoid applied element-wise, values in [0, 1]
    """
    return 1 / (1 + np.exp(-Z))


def relu(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the ReLU activation function element-wise.
    Args:
        Z: Input array of any shape with floating-point values
    Returns:
        Array of the same shape with ReLU applied element-wise, max(0, Z)
    """
    return np.maximum(0, Z)


def softmax(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the softmax activation function row-wise.
    Args:
        Z: Input array of shape (n_samples, n_classes) with floating-point values
    Returns:
        Array of the same shape with softmax applied row-wise, probabilities summing to 1 per row
    """
    # Subtract the max for numerical stability (avoid overflow in exp)
    Z_max = np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z - Z_max)
    sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
    return exp_Z / sum_exp_Z


def mse_loss(y_pred: NDArray[np.floating], y: NDArray[np.floating]) -> np.floating:
    """
    Compute the Mean Squared Error loss between predicted and true values.
    Args:
        y_pred: Predicted values, array of shape (n,) or (n,1) with floating-point values
        y: True values, array of shape (n,) or (n,1) with floating-point values
    Returns:
        Mean squared error as a single float
    """
    return np.mean((y_pred - y) ** 2)


def binary_cross_entropy(
    A: NDArray[np.floating], y: NDArray[np.floating]
) -> np.floating:
    """
    Compute the Binary Cross-Entropy loss between predicted probabilities and true labels.
    Args:
        A: Predicted probabilities (after sigmoid), array of shape (n,) or (n,1), values in [0, 1]
        y: True binary labels, array of shape (n,) or (n,1), values in {0, 1}
    Returns:
        Binary cross-entropy loss as a single float
    """
    # Add small epsilon to avoid log(0) issues
    epsilon = 1e-15
    return -np.mean(y * np.log(A + epsilon) + (1 - y) * np.log(1 - A + epsilon))


def cross_entropy(A: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """
    Compute categorical cross-entropy loss for multi-class classification.
    Args:
        A: Predicted probabilities after softmax, shape (n_samples, n_classes)
        y: True labels, one-hot encoded, shape (n_samples, n_classes)
    Returns:
        Cross-entropy loss as a single float
    """
    epsilon = 1e-15  # Small value to prevent log(0)
    return -np.mean(np.sum(y * np.log(A + epsilon), axis=1))


def gradient_descent(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    W: NDArray[np.floating],
    b: NDArray[np.floating],
    lr: float,
    num_epochs: int,
    batch_size: int,
    loss_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], np.floating],
    activation_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]] = lambda x: x,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], List[float]]:
    """
    Perform mini-batch gradient descent to minimize loss.
    Args:
        X: Input data, shape (n_samples, n_features)
        y: True values, shape (n_samples, 1)
        W: Initial weights, shape (n_features, 1)
        b: Initial bias, shape (1,) or (1,1)
        lr: Learning rate, step size for updates
        num_epochs: Number of full passes through the dataset
        batch_size: Size of each mini-batch
        loss_fn: Loss function to compute error, e.g., mse_loss or binary_cross_entropy
        activation_fn: Activation function to apply to linear output (default: identity)
    Returns:
        Tuple of (updated W, updated b, list of loss values over epochs)
    """
    n_samples = X.shape[0]
    loss_history = []

    for epoch in range(num_epochs):
        # print(f"Epoch {epoch+1}/{num_epochs}")
        # Shuffle the dataset to ensure random mini-batches
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Process mini-batches
        for start_idx in range(0, n_samples, batch_size):
            # print(f"Processing batch starting at index {start_idx}")
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            batch_size_actual = X_batch.shape[0]

            # Forward pass: Compute linear output and apply activation
            Z_batch = X_batch @ W + b
            y_pred_batch = activation_fn(Z_batch)
            # Compute gradients (works for MSE with identity or BCE with sigmoid)
            error = y_pred_batch - y_batch
            grad_W = (X_batch.T @ error) / batch_size_actual
            grad_b = np.mean(error)
            # Update parameters
            W = W - lr * grad_W
            b = b - lr * grad_b

        # Compute loss on full dataset at end of epoch
        y_pred_full = activation_fn(X @ W + b)
        loss = loss_fn(y_pred_full, y)
        loss_history.append(loss)
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    return W, b, loss_history


def numerical_gradient(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    params: Dict[str, NDArray[np.floating]],
    loss_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], np.floating],
    forward_fn: Callable[
        [NDArray[np.floating], Dict[str, NDArray[np.floating]]], NDArray[np.floating]
    ],
    h: float = 1e-4,
) -> Dict[str, NDArray[np.floating]]:
    """
    Compute numerical gradients for parameters using central difference approximation.
    Args:
        X: Input data, shape (n_samples, n_features)
        y: True values, shape (n_samples, 1)
        params: Dictionary of parameters (e.g., {'W': ..., 'b': ...})
        loss_fn: Loss function to compute error, e.g., mse_loss
        forward_fn: Function to compute predictions from X and params
        h: Step size for finite difference approximation (default: 1e-4)
    Returns:
        Dictionary of numerical gradients for each parameter
    """
    num_grads = {}

    for param_name, param_value in params.items():
        num_grad = np.zeros_like(param_value)
        it = np.nditer(param_value, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            original_value = param_value[idx]

            # Compute loss at W + h
            param_value[idx] = original_value + h
            y_pred_plus = forward_fn(X, params)
            loss_plus = loss_fn(y_pred_plus, y)

            # Compute loss at W - h
            param_value[idx] = original_value - h
            y_pred_minus = forward_fn(X, params)
            loss_minus = loss_fn(y_pred_minus, y)

            # Central difference approximation
            num_grad[idx] = (loss_plus - loss_minus) / (2 * h)

            # Restore original value
            param_value[idx] = original_value
            it.iternext()

        num_grads[param_name] = num_grad

    return num_grads


def forward_perceptron(
    X: NDArray[np.floating], W: NDArray[np.floating], b: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Compute the forward pass of a single-layer perceptron.
    Args:
        X: Input data, shape (n_samples, n_features)
        W: Weights, shape (n_features, 1)
        b: Bias, shape (1, 1) or (1,)
    Returns:
        Output after sigmoid activation, shape (n_samples, 1)
    """
    Z = X @ W + b  # Linear combination
    A = sigmoid(Z)  # Sigmoid activation
    return A


def forward_mlp(
    X: NDArray[np.floating],
    W1: NDArray[np.floating],
    b1: NDArray[np.floating],
    W2: NDArray[np.floating],
    b2: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute the forward pass of a 2-layer MLP.
    Args:
        X: Input data, shape (n_samples, n_features, e.g., 784 for MNIST)
        W1: Weights for first layer, shape (n_features, n_hidden, e.g., 784x256)
        b1: Bias for first layer, shape (1, n_hidden)
        W2: Weights for second layer, shape (n_hidden, n_classes, e.g., 256x10)
        b2: Bias for second layer, shape (1, n_classes)
    Returns:
        Tuple of (A1, A2):
        - A1: Hidden layer output after ReLU, shape (n_samples, n_hidden)
        - A2: Output layer output after softmax, shape (n_samples, n_classes)
    """
    Z1 = X @ W1 + b1  # First layer linear combination
    A1 = relu(Z1)  # ReLU activation for hidden layer
    Z2 = A1 @ W2 + b2  # Second layer linear combination
    A2 = softmax(Z2)  # Softmax activation for output layer
    return A1, A2


def backward_mlp(
    X: NDArray[np.floating],
    A1: NDArray[np.floating],
    A2: NDArray[np.floating],
    y: NDArray[np.floating],
    W1: NDArray[np.floating],
    W2: NDArray[np.floating],
    Z1: NDArray[np.floating],
) -> Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Compute gradients for a 2-layer MLP using backpropagation.
    Args:
        X: Input data, shape (n_samples, n_features)
        A1: Hidden layer output after ReLU, shape (n_samples, n_hidden)
        A2: Output layer output after softmax, shape (n_samples, n_classes)
        y: True labels, one-hot encoded, shape (n_samples, n_classes)
        W1: Weights for first layer, shape (n_features, n_hidden)
        W2: Weights for second layer, shape (n_hidden, n_classes)
        Z1: Pre-activation values for hidden layer, shape (n_samples, n_hidden)
    Returns:
        Tuple of gradients (grad_W1, grad_b1, grad_W2, grad_b2)
    """
    n = X.shape[0]

    # Output layer error (delta2)
    delta2 = A2 - y  # Shape (n_samples, n_classes)

    # Gradients for output layer (W2, b2)
    grad_W2 = (A1.T @ delta2) / n  # Shape (n_hidden, n_classes)
    grad_b2 = np.mean(delta2, axis=0, keepdims=True)  # Shape (1, n_classes)

    # Hidden layer error (delta1)
    delta1 = (delta2 @ W2.T) * (Z1 > 0)  # ReLU derivative: 1 if Z1 > 0, 0 otherwise
    # Shape (n_samples, n_hidden)

    # Gradients for hidden layer (W1, b1)
    grad_W1 = (X.T @ delta1) / n  # Shape (n_features, n_hidden)
    grad_b1 = np.mean(delta1, axis=0, keepdims=True)  # Shape (1, n_hidden)

    return grad_W1, grad_b1, grad_W2, grad_b2


def forward_mlp_3layer(
    X: NDArray[np.floating],
    W1: NDArray[np.floating],
    b1: NDArray[np.floating],
    W2: NDArray[np.floating],
    b2: NDArray[np.floating],
    W3: NDArray[np.floating],
    b3: NDArray[np.floating],
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute the forward pass of a 3-layer MLP.
    Args:
        X: Input data, shape (n_samples, n_features, e.g., 784 for MNIST)
        W1: Weights for first layer, shape (n_features, n_hidden1, e.g., 784x256)
        b1: Bias for first layer, shape (1, n_hidden1)
        W2: Weights for second layer, shape (n_hidden1, n_hidden2, e.g., 256x128)
        b2: Bias for second layer, shape (1, n_hidden2)
        W3: Weights for third layer, shape (n_hidden2, n_classes, e.g., 128x10)
        b3: Bias for third layer, shape (1, n_classes)
    Returns:
        Tuple of (A1, A2, A3):
        - A1: First hidden layer output after ReLU, shape (n_samples, n_hidden1)
        - A2: Second hidden layer output after ReLU, shape (n_samples, n_hidden2)
        - A3: Output layer output after softmax, shape (n_samples, n_classes)
    """
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)
    return A1, A2, A3


def backward_mlp_3layer(
    X: NDArray[np.floating],
    A1: NDArray[np.floating],
    A2: NDArray[np.floating],
    A3: NDArray[np.floating],
    y: NDArray[np.floating],
    W1: NDArray[np.floating],
    W2: NDArray[np.floating],
    W3: NDArray[np.floating],
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
) -> Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Compute gradients for a 3-layer MLP using backpropagation.
    Args:
        X: Input data, shape (n_samples, n_features)
        A1: First hidden layer output after ReLU, shape (n_samples, n_hidden1)
        A2: Second hidden layer output after ReLU, shape (n_samples, n_hidden2)
        A3: Output layer output after softmax, shape (n_samples, n_classes)
        y: True labels, one-hot encoded, shape (n_samples, n_classes)
        W1: Weights for first layer, shape (n_features, n_hidden1)
        W2: Weights for second layer, shape (n_hidden1, n_hidden2)
        W3: Weights for third layer, shape (n_hidden2, n_classes)
        Z1: Pre-activation values for first hidden layer, shape (n_samples, n_hidden1)
        Z2: Pre-activation values for second hidden layer, shape (n_samples, n_hidden2)
    Returns:
        Tuple of gradients (grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3)
    """
    n = X.shape[0]

    # Output layer error (delta3)
    delta3 = A3 - y  # Shape (n_samples, n_classes)

    # Gradients for output layer (W3, b3)
    grad_W3 = (A2.T @ delta3) / n  # Shape (n_hidden2, n_classes)
    grad_b3 = np.mean(delta3, axis=0, keepdims=True)  # Shape (1, n_classes)

    # Second hidden layer error (delta2)
    delta2 = (delta3 @ W3.T) * (Z2 > 0)  # ReLU derivative: 1 if Z2 > 0, 0 otherwise
    # Shape (n_samples, n_hidden2)

    # Gradients for second hidden layer (W2, b2)
    grad_W2 = (A1.T @ delta2) / n  # Shape (n_hidden1, n_hidden2)
    grad_b2 = np.mean(delta2, axis=0, keepdims=True)  # Shape (1, n_hidden2)

    # First hidden layer error (delta1)
    delta1 = (delta2 @ W2.T) * (Z1 > 0)  # ReLU derivative: 1 if Z1 > 0, 0 otherwise
    # Shape (n_samples, n_hidden1)

    # Gradients for first hidden layer (W1, b1)
    grad_W1 = (X.T @ delta1) / n  # Shape (n_features, n_hidden1)
    grad_b1 = np.mean(delta1, axis=0, keepdims=True)  # Shape (1, n_hidden1)

    return grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3


def conv2d(
    image: NDArray[np.floating], filter_kernel: NDArray[np.floating], stride: int = 1
) -> NDArray[np.floating]:
    """
    Perform 2D convolution on an image using a filter kernel.
    Args:
        image: Input image, shape (height, width)
        filter_kernel: Convolution filter, shape (filter_height, filter_width)
        stride: Stride of the convolution operation (default: 1)
    Returns:
        Output feature map after convolution, shape depends on input, filter size, and stride
    """
    # Use scipy.signal.convolve2d with 'valid' mode (no padding)
    # 'valid' mode means output size is reduced based on filter size
    output = signal.convolve2d(
        image, filter_kernel, mode="valid", boundary="fill", fillvalue=0
    )

    # Apply stride by downsampling the output
    if stride > 1:
        output = output[::stride, ::stride]

    return output


def max_pool(
    X: NDArray[np.floating], size: int = 2, stride: int = 2
) -> NDArray[np.floating]:
    """
    Perform 2D max pooling on an input feature map.
    Args:
        X: Input feature map, shape (height, width) or (height, width, channels)
        size: Size of the pooling window (default: 2 for 2x2 pooling)
        stride: Stride of the pooling operation (default: 2)
    Returns:
        Output after max pooling, shape depends on input, size, and stride
    """
    if len(X.shape) == 2:
        height, width = X.shape
        channels = 1
        X = X.reshape(height, width, 1)
    else:
        height, width, channels = X.shape

    out_height = (height - size) // stride + 1
    out_width = (width - size) // stride + 1
    output = np.zeros((out_height, out_width, channels))

    for i in range(out_height):
        for j in range(out_width):
            x_start = i * stride
            x_end = x_start + size
            y_start = j * stride
            y_end = y_start + size
            region = X[x_start:x_end, y_start:y_end, :]
            output[i, j, :] = np.max(region, axis=(0, 1))

    if channels == 1:
        output = output[:, :, 0]
    return output


def l2_regularization(
    weights: List[NDArray[np.floating]], lambda_: float
) -> tuple[float, List[NDArray[np.floating]]]:
    """
    Compute L2 regularization penalty and gradients for a list of weight matrices.
    Args:
        weights: List of weight matrices (e.g., [W1, W2, W3])
        lambda_: Regularization strength (e.g., 0.01)
    Returns:
        Tuple of (l2_penalty, l2_grads):
        - l2_penalty: Scalar penalty term to add to loss (lambda * sum of squared weights)
        - l2_grads: List of gradients for each weight matrix (2 * lambda * W)
    """
    l2_penalty = 0.0
    l2_grads = []
    for W in weights:
        l2_penalty += np.sum(W**2)
        l2_grads.append(2 * lambda_ * W)
    l2_penalty *= lambda_
    l2_penalty = cast(float, l2_penalty)  # Ensure penalty is a scalar
    return l2_penalty, l2_grads


def dropout(
    A: NDArray[np.floating], p: float, training: bool = True
) -> NDArray[np.floating]:
    """
    Apply dropout to an activation matrix by randomly setting elements to 0.
    Args:
        A: Activation matrix, shape (any)
        p: Keep probability (e.g., 0.8 to keep 80% of neurons)
        training: Boolean, apply dropout only during training (default: True)
    Returns:
        Activation matrix after dropout (same shape as input)
    """
    if training:
        mask = np.random.binomial(1, p, size=A.shape)
        return A * mask
    return A


def momentum_update(
    velocity: NDArray[np.floating], gradient: NDArray[np.floating], mu: float, lr: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Perform a momentum-based update on the velocity and return the parameter update.
    Args:
        velocity: Current velocity array, same shape as gradient (initialized to 0)
        gradient: Current gradient of the loss with respect to the parameter
        mu: Momentum coefficient (e.g., 0.9)
        lr: Learning rate (e.g., 0.1)
    Returns:
        Tuple of (updated_velocity, parameter_update):
        - updated_velocity: New velocity after momentum update
        - parameter_update: Update to apply to the parameter (e.g., W += parameter_update)
    """
    updated_velocity = mu * velocity - lr * gradient
    parameter_update = updated_velocity
    return updated_velocity, parameter_update


def accuracy(y_pred: NDArray[np.floating], y_true: NDArray[np.floating]) -> float:
    """
    Compute classification accuracy for multi-class predictions.
    Args:
        y_pred: Predicted probabilities or logits, shape (n_samples, n_classes)
        y_true: True labels, one-hot encoded or class indices, shape (n_samples, n_classes) or (n_samples,)
    Returns:
        Accuracy as a float (fraction of correct predictions)
    """
    if y_true.ndim == 2:  # One-hot encoded
        true_labels = np.argmax(y_true, axis=1)
    else:  # Class indices
        true_labels = y_true
    pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(pred_labels == true_labels)
