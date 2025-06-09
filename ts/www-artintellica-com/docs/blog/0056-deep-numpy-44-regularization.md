+++
title = "Learn Deep Learning with NumPy, Part 4.4: Regularization Techniques"
author = "Artintellica"
date = "2025-06-09"
+++

# Learn Deep Learning with NumPy, Part 4.4: Regularization Techniques

## Introduction

Welcome back to our blog series, _"Learn Deep Learning with NumPy"_! In Part 4.3, we introduced pooling layers and constructed a simple CNN architecture for MNIST, leveraging spatial hierarchies for efficient image processing. Now, in Part 4.4, we’ll address a critical challenge in training deep models: *overfitting*. We’ll implement two powerful regularization techniques—L2 regularization and dropout—to prevent overfitting and improve generalization on unseen data.

By the end of this post, you’ll understand the mathematics behind L2 regularization and dropout, implement `l2_regularization()` and `dropout()` in NumPy, and apply them to train a 3-layer MLP on MNIST, comparing accuracy with and without regularization. These techniques enhance our training process, reusing `gradient_descent()` and other toolkit functions. Let’s dive into the math and code for regularization!

---

## Why Regularization Techniques Matter in Deep Learning

Deep neural networks, with their large number of parameters, are prone to *overfitting*—memorizing the training data rather than learning general patterns. This results in high accuracy on training data but poor performance on unseen test data. *Regularization techniques* combat overfitting by adding constraints or randomness to the model, encouraging simpler solutions or reducing reliance on specific neurons.

In deep learning, regularization is crucial because:
- It prevents models from fitting noise in the training data, improving generalization.
- It allows deeper or more complex models to be trained without catastrophic overfitting.
- It stabilizes training, especially in the presence of limited data.

In this post, we’ll focus on two popular regularization methods:
- **L2 Regularization**: Adds a penalty on the magnitude of weights, encouraging smaller weights and smoother decision boundaries.
- **Dropout**: Randomly deactivates neurons during training, preventing co-dependency and promoting redundancy.

We’ll apply these to a 3-layer MLP on MNIST to observe their impact on training and test accuracy. Let’s explore the math behind these techniques.

---

## Mathematical Foundations: L2 Regularization and Dropout

### L2 Regularization

L2 regularization, also known as weight decay, adds a penalty term to the loss function based on the squared magnitude of the weights. For a loss function $L_{\text{data}}$ (e.g., cross-entropy), the regularized loss becomes:

$$
L = L_{\text{data}} + \lambda \sum W^2
$$

Where:
- $\sum W^2$ is the sum of squared weights across all layers (L2 norm squared).
- $\lambda$ (lambda) is a hyperparameter controlling the strength of regularization (e.g., 0.01).
- This penalty discourages large weights, promoting simpler models less likely to overfit.

During backpropagation, the gradient of the loss with respect to a weight $W$ includes an additional term:

$$
\nabla_W L = \nabla_W L_{\text{data}} + 2\lambda W
$$

This effectively "decays" weights during updates, as the update rule becomes:

$$
W \leftarrow W - \eta (\nabla_W L_{\text{data}} + 2\lambda W)
$$

Where $\eta$ is the learning rate. This can also be implemented as a separate penalty added to gradients or by scaling weights after updates.

### Dropout

Dropout is a technique that randomly "drops" (deactivates) a fraction of neurons during training, forcing the network to learn redundant representations. For an activation matrix $A$ (e.g., output of a layer), dropout applies a random mask:

$$
A_{\text{drop}} = A \cdot \text{mask}, \quad \text{mask} \sim \text{Bernoulli}(p)
$$

Where:
- $\text{mask}$ is a binary matrix of the same shape as $A$, with each element drawn from a Bernoulli distribution with probability $p$ (keep probability, e.g., 0.8 means 80% of neurons are kept, 20% dropped).
- During training, dropped neurons (mask=0) do not contribute to forward or backward passes.
- At test time, dropout is disabled, and activations are often scaled by $p$ to maintain expected output magnitude (though we’ll simplify by not scaling in our implementation for now).

Dropout reduces overfitting by preventing neurons from overly specializing or co-adapting, acting as a form of ensemble averaging over many sub-networks. Now, let’s implement these regularization techniques in NumPy and apply them to MNIST training.

---

## Implementing Regularization Techniques with NumPy

We’ll create `l2_regularization()` to compute the L2 penalty and its gradient, and `dropout()` to randomly deactivate neurons during training. We’ll then train a 3-layer MLP on MNIST with these techniques, comparing performance to a baseline without regularization.

### L2 Regularization Implementation

Here’s the implementation of L2 regularization to compute the penalty term and its gradient contribution:

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union, List

def l2_regularization(weights: List[NDArray[np.floating]], lambda_: float) -> tuple[float, List[NDArray[np.floating]]]:
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
        l2_penalty += np.sum(W ** 2)
        l2_grads.append(2 * lambda_ * W)
    l2_penalty *= lambda_
    return l2_penalty, l2_grads
```

### Dropout Implementation

Here’s the implementation of dropout to randomly deactivate neurons during training:

```python
def dropout(A: NDArray[np.floating], p: float, training: bool = True) -> NDArray[np.floating]:
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
```

### Example: Training a 3-Layer MLP with Regularization on MNIST

Let’s train a 3-layer MLP (784 → 256 → 128 → 10) on MNIST with L2 regularization and dropout, comparing performance to a baseline without regularization. We’ll visualize loss and accuracy over epochs using `matplotlib`.

**Note**: Ensure you have `sklearn` (`pip install scikit-learn`) for loading MNIST and `matplotlib` (`pip install matplotlib`) for plotting. We’ll use a subset of MNIST for CPU efficiency.

```python
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_network import normalize, relu, softmax, cross_entropy, forward_mlp_3layer, backward_mlp_3layer

# Load MNIST data (subset for faster training on CPU)
print("Loading MNIST data...")
X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X_full = X_full.astype(float)
y_full = y_full.astype(int)

# Limit to 5000 samples for faster training
X = X_full[:5000]
y = y_full[:5000]

# Convert labels to one-hot encoding
n_classes = 10
y_one_hot = np.zeros((y.shape[0], n_classes))
y_one_hot[np.arange(y.shape[0]), y] = 1

# Normalize input data
X = normalize(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Initialize parameters for 3-layer MLP (784 -> 256 -> 128 -> 10)
n_features = X_train.shape[1]  # 784 for MNIST
n_hidden1 = 256
n_hidden2 = 128
W1 = np.random.randn(n_features, n_hidden1) * 0.01
b1 = np.zeros((1, n_hidden1))
W2 = np.random.randn(n_hidden1, n_hidden2) * 0.01
b2 = np.zeros((1, n_hidden2))
W3 = np.random.randn(n_hidden2, n_classes) * 0.01
b3 = np.zeros((1, n_classes))

# Training loop with L2 regularization and dropout
lr = 0.1
num_epochs = 20
batch_size = 64
lambda_l2 = 0.01  # L2 regularization strength
dropout_p = 0.8   # Keep probability for dropout (80% keep, 20% drop)
n_samples = X_train.shape[0]
loss_history_with_reg = []
accuracy_history_with_reg = []
loss_history_no_reg = []
accuracy_history_no_reg = []

# Function to train with or without regularization
def train_mlp(with_reg: bool, W1, b1, W2, b2, W3, b3):
    W1_copy = W1.copy()
    b1_copy = b1.copy()
    W2_copy = W2.copy()
    b2_copy = b2.copy()
    W3_copy = W3.copy()
    b3_copy = b3.copy()
    loss_history = []
    accuracy_history = []
    for epoch in range(num_epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        total_loss = 0.0
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            # Forward pass with dropout if enabled
            A1, A2, A3 = forward_mlp_3layer(X_batch, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy)
            if with_reg:
                A1 = dropout(A1, dropout_p, training=True)
                A2 = dropout(A2, dropout_p, training=True)
            # Compute loss
            data_loss = cross_entropy(A3, y_batch)
            l2_penalty = 0.0
            l2_grads = []
            if with_reg:
                l2_penalty, l2_grads = l2_regularization([W1_copy, W2_copy, W3_copy], lambda_l2)
            total_loss += data_loss + l2_penalty
            # Backpropagation
            Z1 = X_batch @ W1_copy + b1_copy
            Z2 = A1 @ W2_copy + b2_copy
            grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = backward_mlp_3layer(
                X_batch, A1, A2, A3, y_batch, W1_copy, W2_copy, W3_copy, Z1, Z2)
            if with_reg:
                grad_W1 += l2_grads[0]
                grad_W2 += l2_grads[1]
                grad_W3 += l2_grads[2]
            # Update parameters
            W1_copy -= lr * grad_W1
            b1_copy -= lr * grad_b1
            W2_copy -= lr * grad_W2
            b2_copy -= lr * grad_b2
            W3_copy -= lr * grad_W3
            b3_copy -= lr * grad_b3
        # Compute loss on full training set (without dropout)
        _, _, A3_full = forward_mlp_3layer(X_train, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy)
        loss = cross_entropy(A3_full, y_train)
        loss_history.append(loss)
        # Compute accuracy on test set (without dropout)
        _, _, A3_test = forward_mlp_3layer(X_test, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy)
        predictions = np.argmax(A3_test, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == true_labels)
        accuracy_history.append(accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f} (With Reg: {with_reg})")
    return loss_history, accuracy_history

# Train with regularization
print("Training with L2 Regularization and Dropout...")
loss_history_with_reg, accuracy_history_with_reg = train_mlp(
    True, W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy())

# Train without regularization
print("Training without Regularization...")
loss_history_no_reg, accuracy_history_no_reg = train_mlp(
    False, W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy())

# Plot loss and accuracy history for comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.plot(range(1, num_epochs + 1), loss_history_with_reg, label='Training Loss (With Reg)')
ax1.plot(range(1, num_epochs + 1), loss_history_no_reg, label='Training Loss (No Reg)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.set_title('Training Loss Over Epochs')
ax1.legend()
ax1.grid(True)

ax2.plot(range(1, num_epochs + 1), accuracy_history_with_reg, label='Test Accuracy (With Reg)')
ax2.plot(range(1, num_epochs + 1), accuracy_history_no_reg, label='Test Accuracy (No Reg)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Test Accuracy Over Epochs')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

**Output** (approximate, values will vary due to randomness and subset size):
```
Loading MNIST data...
Training with L2 Regularization and Dropout...
Epoch 1/20, Loss: 2.3015, Test Accuracy: 0.1100 (With Reg: True)
...
Epoch 20/20, Loss: 0.3652, Test Accuracy: 0.8850 (With Reg: True)
Training without Regularization...
Epoch 1/20, Loss: 2.3018, Test Accuracy: 0.1080 (With Reg: False)
...
Epoch 20/20, Loss: 0.3127, Test Accuracy: 0.8700 (With Reg: False)
```

**Loss and Accuracy Plots**: (Two `matplotlib` plots will display, comparing training loss and test accuracy over epochs for models with and without regularization. Typically, the model with regularization shows slightly higher training loss but better test accuracy due to reduced overfitting.)

In this example, we train a 3-layer MLP (784 → 256 → 128 → 10) on a subset of MNIST (5000 samples for CPU efficiency) over 20 epochs with batch size 64. We compare two setups: one with L2 regularization (`lambda_=0.01`) and dropout (`p=0.8`), and one without. The model with regularization often achieves slightly better test accuracy (~88-90% vs. ~85-87%) due to reduced overfitting, though training loss might be higher. Training time remains manageable (~3-6 minutes on CPU).

---

## Organizing Our Growing Library

Let’s update our `neural_network.py` file to include the `l2_regularization()` and `dropout()` functions alongside our previous implementations. These will enhance our ability to train deep models without overfitting.

```python
# neural_network.py
import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable, Tuple, List, Dict
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
    if np.any(mask):
        normalized_X[:, mask] = normalized_X[:, mask] / std[mask]
    return normalized_X

def matrix_multiply(X: NDArray[np.floating], W: NDArray[np.floating]) -> NDArray[np.floating]:
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
    Z_max = np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z - Z_max)
    sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
    return exp_Z / sum_exp_Z

def mse_loss(y_pred: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """
    Compute the Mean Squared Error loss between predicted and true values.
    Args:
        y_pred: Predicted values, array of shape (n,) or (n,1) with floating-point values
        y: True values, array of shape (n,) or (n,1) with floating-point values
    Returns:
        Mean squared error as a single float
    """
    return np.mean((y_pred - y) ** 2)

def binary_cross_entropy(A: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """
    Compute the Binary Cross-Entropy loss between predicted probabilities and true labels.
    Args:
        A: Predicted probabilities (after sigmoid), array of shape (n,) or (n,1), values in [0, 1]
        y: True binary labels, array of shape (n,) or (n,1), values in {0, 1}
    Returns:
        Binary cross-entropy loss as a single float
    """
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

def gradient_descent(X: NDArray[np.floating], y: NDArray[np.floating], W: NDArray[np.floating], 
                     b: NDArray[np.floating], lr: float, num_epochs: int, batch_size: int, 
                     loss_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], float],
                     activation_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]] = lambda x: x) -> Tuple[NDArray[np.floating], NDArray[np.floating], List[float]]:
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
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            batch_size_actual = X_batch.shape[0]
            
            Z_batch = X_batch @ W + b
            y_pred_batch = activation_fn(Z_batch)
            error = y_pred_batch - y_batch
            grad_W = (X_batch.T @ error) / batch_size_actual
            grad_b = np.mean(error)
            W = W - lr * grad_W
            b = b - lr * grad_b
        
        y_pred_full = activation_fn(X @ W + b)
        loss = loss_fn(y_pred_full, y)
        loss_history.append(loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    return W, b, loss_history

def numerical_gradient(X: NDArray[np.floating], y: NDArray[np.floating], params: Dict[str, NDArray[np.floating]], 
                      loss_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], float], 
                      forward_fn: Callable[[NDArray[np.floating], Dict[str, NDArray[np.floating]]], NDArray[np.floating]], 
                      h: float = 1e-4) -> Dict[str, NDArray[np.floating]]:
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
        it = np.nditer(param_value, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            original_value = param_value[idx]
            
            param_value[idx] = original_value + h
            y_pred_plus = forward_fn(X, params)
            loss_plus = loss_fn(y_pred_plus, y)
            
            param_value[idx] = original_value - h
            y_pred_minus = forward_fn(X, params)
            loss_minus = loss_fn(y_pred_minus, y)
            
            num_grad[idx] = (loss_plus - loss_minus) / (2 * h)
            
            param_value[idx] = original_value
            it.iternext()
        
        num_grads[param_name] = num_grad
    
    return num_grads

def forward_perceptron(X: NDArray[np.floating], W: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.floating]:
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

def forward_mlp(X: NDArray[np.floating], W1: NDArray[np.floating], b1: NDArray[np.floating], 
                W2: NDArray[np.floating], b2: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
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
    A1 = relu(Z1)      # ReLU activation for hidden layer
    Z2 = A1 @ W2 + b2  # Second layer linear combination
    A2 = softmax(Z2)   # Softmax activation for output layer
    return A1, A2

def backward_mlp(X: NDArray[np.floating], A1: NDArray[np.floating], A2: NDArray[np.floating], 
                y: NDArray[np.floating], W1: NDArray[np.floating], W2: NDArray[np.floating], 
                Z1: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
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

def forward_mlp_3layer(X: NDArray[np.floating], W1: NDArray[np.floating], b1: NDArray[np.floating], 
                       W2: NDArray[np.floating], b2: NDArray[np.floating], 
                       W3: NDArray[np.floating], b3: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
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

def backward_mlp_3layer(X: NDArray[np.floating], A1: NDArray[np.floating], A2: NDArray[np.floating], 
                        A3: NDArray[np.floating], y: NDArray[np.floating], 
                        W1: NDArray[np.floating], W2: NDArray[np.floating], W3: NDArray[np.floating], 
                        Z1: NDArray[np.floating], Z2: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
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

def conv2d(image: NDArray[np.floating], filter_kernel: NDArray[np.floating], stride: int = 1) -> NDArray[np.floating]:
    """
    Perform 2D convolution on an image using a filter kernel.
    Args:
        image: Input image, shape (height, width)
        filter_kernel: Convolution filter, shape (filter_height, filter_width)
        stride: Stride of the convolution operation (default: 1)
    Returns:
        Output feature map after convolution, shape depends on input, filter size, and stride
    """
    output = signal.convolve2d(image, filter_kernel, mode='valid', boundary='fill', fillvalue=0)
    if stride > 1:
        output = output[::stride, ::stride]
    return output

def max_pool(X: NDArray[np.floating], size: int = 2, stride: int = 2) -> NDArray[np.floating]:
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

def l2_regularization(weights: List[NDArray[np.floating]], lambda_: float) -> Tuple[float, List[NDArray[np.floating]]]:
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
        l2_penalty += np.sum(W ** 2)
        l2_grads.append(2 * lambda_ * W)
    l2_penalty *= lambda_
    return l2_penalty, l2_grads

def dropout(A: NDArray[np.floating], p: float, training: bool = True) -> NDArray[np.floating]:
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
```

You can now import these new functions using `from neural_network import l2_regularization, dropout`. They enhance our training capabilities by adding regularization to prevent overfitting in deep models.

---

## Exercises: Practice with Regularization Techniques

To reinforce your understanding of L2 regularization and dropout, try these Python-focused coding exercises. They’ll help you build intuition for how regularization prevents overfitting and impacts model performance. Run the code and compare outputs to verify your solutions.

1. **L2 Regularization Penalty Calculation**  
   Create two small weight matrices `W1 = np.array([[1.0, 2.0], [3.0, 4.0]])` and `W2 = np.array([[0.5, 1.5], [2.5, 3.5]])`. Compute the L2 regularization penalty and gradients using `l2_regularization()` with `lambda_=0.01`. Print the penalty and gradients, and verify that the penalty is `lambda_ * sum of squared weights` and gradients are `2 * lambda_ * W`.

   ```python
   # Your code here
   W1 = np.array([[1.0, 2.0], [3.0, 4.0]])
   W2 = np.array([[0.5, 1.5], [2.5, 3.5]])
   lambda_ = 0.01
   l2_penalty, l2_grads = l2_regularization([W1, W2], lambda_)
   print("L2 Penalty:", l2_penalty)
   print("Gradient for W1:\n", l2_grads[0])
   print("Gradient for W2:\n", l2_grads[1])
   ```

2. **Dropout on Synthetic Activations**  
   Create a synthetic activation matrix `A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])`. Apply `dropout()` with `p=0.5` (50% keep probability) during training (`training=True`). Run it twice to observe randomness in dropped elements (values set to 0). Then, apply with `training=False` to verify no dropout occurs. Print results each time.

   ```python
   # Your code here
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
   ```

3. **L2 Regularization in Single Epoch Training**  
   Use a small synthetic dataset `X = np.random.randn(5, 3)` (5 samples, 3 features) and one-hot labels `y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]])` (2 classes). Initialize a 3 → 4 → 2 MLP with small random weights. Perform one epoch of training with `forward_mlp()` and `backward_mlp()`, adding L2 regularization (`lambda_=0.01`). Compare the loss with and without L2 penalty. Print both losses.

   ```python
   # Your code here
   X = np.random.randn(5, 3)
   y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]])
   W1 = np.random.randn(3, 4) * 0.1
   b1 = np.zeros((1, 4))
   W2 = np.random.randn(4, 2) * 0.1
   b2 = np.zeros((1, 2))
   lr = 0.1
   lambda_ = 0.01
   # Forward pass
   Z1 = X @ W1 + b1
   A1 = relu(Z1)
   Z2 = A1 @ W2 + b2
   A2 = softmax(Z2)
   # Loss without regularization
   data_loss = cross_entropy(A2, y)
   print("Data Loss (without L2):", data_loss)
   # Loss with L2 regularization
   l2_penalty, l2_grads = l2_regularization([W1, W2], lambda_)
   total_loss = data_loss + l2_penalty
   print("Total Loss (with L2):", total_loss)
   # Backpropagation with L2 gradients
   delta2 = A2 - y
   grad_W2 = (A1.T @ delta2) / X.shape[0] + l2_grads[1]
   grad_b2 = np.mean(delta2, axis=0, keepdims=True)
   delta1 = (delta2 @ W2.T) * (Z1 > 0)
   grad_W1 = (X.T @ delta1) / X.shape[0] + l2_grads[0]
   grad_b1 = np.mean(delta1, axis=0, keepdims=True)
   # Update parameters
   W1 -= lr * grad_W1
   b1 -= lr * grad_b1
   W2 -= lr * grad_W2
   b2 -= lr * grad_b2
   ```

4. **Training with Dropout on Small MNIST Subset**  
   Load a small subset of MNIST (e.g., 1000 samples) using `fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)`. Initialize a 784 → 256 → 128 → 10 MLP. Train for 10 epochs with `lr = 0.1`, `batch_size = 32`, applying dropout (`p=0.8`) to hidden layers during training (disable during evaluation). Compare test accuracy to a model without dropout. Print results and plot loss histories.

   ```python
   # Your code here
   from sklearn.datasets import fetch_openml
   from sklearn.model_selection import train_test_split
   import matplotlib.pyplot as plt
   X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
   X_full = X_full.astype(float)
   y_full = y_full.astype(int)
   X = X_full[:1000]
   y = y_full[:1000]
   n_classes = 10
   y_one_hot = np.zeros((y.shape[0], n_classes))
   y_one_hot[np.arange(y.shape[0]), y] = 1
   X = normalize(X)
   X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
   W1 = np.random.randn(784, 256) * 0.01
   b1 = np.zeros((1, 256))
   W2 = np.random.randn(256, 128) * 0.01
   b2 = np.zeros((1, 128))
   W3 = np.random.randn(128, 10) * 0.01
   b3 = np.zeros((1, 10))
   lr = 0.1
   num_epochs = 10
   batch_size = 32
   dropout_p = 0.8
   n_samples = X_train.shape[0]
   loss_history_with_dropout = []
   accuracy_history_with_dropout = []
   loss_history_no_dropout = []
   accuracy_history_no_dropout = []
   
   def train_mlp_with_dropout(use_dropout: bool, W1, b1, W2, b2, W3, b3):
       W1_copy = W1.copy()
       b1_copy = b1.copy()
       W2_copy = W2.copy()
       b2_copy = b2.copy()
       W3_copy = W3.copy()
       b3_copy = b3.copy()
       loss_history = []
       accuracy_history = []
       for epoch in range(num_epochs):
           indices = np.random.permutation(n_samples)
           X_shuffled = X_train[indices]
           y_shuffled = y_train[indices]
           for start_idx in range(0, n_samples, batch_size):
               end_idx = min(start_idx + batch_size, n_samples)
               X_batch = X_shuffled[start_idx:end_idx]
               y_batch = y_shuffled[start_idx:end_idx]
               A1, A2, A3 = forward_mlp_3layer(X_batch, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy)
               if use_dropout:
                   A1 = dropout(A1, dropout_p, training=True)
                   A2 = dropout(A2, dropout_p, training=True)
               Z1 = X_batch @ W1_copy + b1_copy
               Z2 = A1 @ W2_copy + b2_copy
               grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = backward_mlp_3layer(
                   X_batch, A1, A2, A3, y_batch, W1_copy, W2_copy, W3_copy, Z1, Z2)
               W1_copy -= lr * grad_W1
               b1_copy -= lr * grad_b1
               W2_copy -= lr * grad_W2
               b2_copy -= lr * grad_b2
               W3_copy -= lr * grad_W3
               b3_copy -= lr * grad_b3
           _, _, A3_full = forward_mlp_3layer(X_train, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy)
           loss = cross_entropy(A3_full, y_train)
           loss_history.append(loss)
           _, _, A3_test = forward_mlp_3layer(X_test, W1_copy, b1_copy, W2_copy, b2_copy, W3_copy, b3_copy)
           predictions = np.argmax(A3_test, axis=1)
           true_labels = np.argmax(y_test, axis=1)
           accuracy = np.mean(predictions == true_labels)
           accuracy_history.append(accuracy)
           print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f} (Dropout: {use_dropout})")
       return loss_history, accuracy_history
   
   print("Training with Dropout...")
   loss_history_with_dropout, accuracy_history_with_dropout = train_mlp_with_dropout(
       True, W1, b1, W2, b2, W3, b3)
   
   print("Training without Dropout...")
   loss_history_no_dropout, accuracy_history_no_dropout = train_mlp_with_dropout(
       False, W1, b1, W2, b2, W3, b3)
   
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
   ax1.plot(range(1, num_epochs + 1), loss_history_with_dropout, label='Training Loss (With Dropout)')
   ax1.plot(range(1, num_epochs + 1), loss_history_no_dropout, label='Training Loss (No Dropout)')
   ax1.set_xlabel('Epoch')
   ax1.set_ylabel('Cross-Entropy Loss')
   ax1.set_title('Training Loss Over Epochs')
   ax1.legend()
   ax1.grid(True)
   
   ax2.plot(range(1, num_epochs + 1), accuracy_history_with_dropout, label='Test Accuracy (With Dropout)')
   ax2.plot(range(1, num_epochs + 1), accuracy_history_no_dropout, label='Test Accuracy (No Dropout)')
   ax2.set_xlabel('Epoch')
   ax2.set_ylabel('Accuracy')
   ax2.set_title('Test Accuracy Over Epochs')
   ax2.legend()
   ax2.grid(True)
   
   plt.tight_layout()
   plt.show()
   ```

These exercises will help you build intuition for implementing L2 regularization and dropout, understanding their impact on loss and gradients, and observing how they improve generalization in deep models like MLPs.

---

## Closing Thoughts

Congratulations on mastering L2 regularization and dropout to combat overfitting! In this post, we’ve explored the mathematics behind these techniques, implemented `l2_regularization()` and `dropout()` in NumPy, and trained a 3-layer MLP on MNIST, demonstrating improved generalization with regularization (often achieving slightly better test accuracy compared to no regularization). This enhances our training capabilities, preparing us for more robust deep learning models.

In the final chapter of our series (Part 4.5: _Training CNNs with Backpropagation_), we’ll implement backpropagation for CNNs, including gradients for convolutional and pooling layers, and train a complete CNN on MNIST to achieve high accuracy, wrapping up our deep learning journey with a powerful image classification model.

Until then, experiment with the code and exercises above. If you have questions or want to share your solutions, drop a comment below—I’m excited to hear from you. Let’s keep building our deep learning toolkit together!

**Next Up**: Part 4.5 – Training CNNs with Backpropagation
