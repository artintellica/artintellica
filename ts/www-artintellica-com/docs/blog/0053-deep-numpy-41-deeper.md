+++
title = "Learn Deep Learning with NumPy, Part 4.1: Deeper MLPs and Vanishing Gradients"
author = "Artintellica"
date = "2025-06-09"
+++

## Introduction

Welcome back to our blog series, _"Learn Deep Learning with NumPy"_! Having completed Module 3, where we built and trained a 2-layer Multi-Layer Perceptron (MLP) on MNIST using forward propagation and backpropagation, we’re now entering the final module: *Deep Learning Challenges and Solutions*. In Part 4.1, we’ll extend our MLP to a deeper 3-layer architecture and explore a key challenge in deep networks—vanishing gradients—which can hinder training as layers increase.

By the end of this post, you’ll understand how to implement forward and backward passes for a 3-layer MLP, recognize the vanishing gradient problem, and train a deeper network on MNIST while visualizing loss and accuracy with `matplotlib`. We’ll reuse and extend our existing functions like `forward_mlp()` and `backward_mlp()`, building on the foundation from Module 3. Let’s dive into the math and code for deeper neural networks!

---

## Why Deeper MLPs and Vanishing Gradients Matter in Deep Learning

Adding more layers to a neural network increases its capacity to learn hierarchical features, enabling it to solve more complex problems. A deeper MLP, such as a 3-layer network, can capture finer patterns in data like MNIST by learning low-level features (e.g., edges) in early layers and combining them into higher-level concepts (e.g., digit shapes) in later layers. However, deeper networks introduce significant challenges, notably the *vanishing gradient problem*.

The vanishing gradient problem occurs during backpropagation when gradients diminish as they are propagated backward through many layers. Small gradients lead to tiny updates to weights in early layers, effectively stalling learning. This often happens with activation functions like sigmoid, whose derivatives are less than 1, causing repeated multiplication to shrink gradients exponentially. ReLU helps mitigate this by maintaining gradient magnitude for positive inputs, but deeper networks can still face issues.

In this post, we’ll implement a 3-layer MLP (784 → 256 → 128 → 10) for MNIST digit classification, extend our forward and backward pass functions, and discuss vanishing gradients while observing training behavior. Let’s explore the math behind a deeper network.

---

## Mathematical Foundations: Forward and Backward Passes in a 3-Layer MLP

### Forward Propagation

Forward propagation in a 3-layer MLP processes input data through three successive transformations to produce predictions. For an input matrix $X$ of shape $(n, d)$ (where $n$ is the number of samples and $d$ is the input features, e.g., 784 for MNIST), the steps are:

1. **First Layer (Hidden Layer 1)**:
   - Linear combination:
     $$
     Z_1 = X W_1 + b_1
     $$
     Where $W_1$ is shape $(d, h_1)$ ($h_1$ is first hidden layer size, e.g., 256), $b_1$ is shape $(1, h_1)$, and $Z_1$ is shape $(n, h_1)$.
   - ReLU activation:
     $$
     A_1 = \text{ReLU}(Z_1) = \max(0, Z_1)
     $$

2. **Second Layer (Hidden Layer 2)**:
   - Linear combination:
     $$
     Z_2 = A_1 W_2 + b_2
     $$
     Where $W_2$ is shape $(h_1, h_2)$ ($h_2$ is second hidden layer size, e.g., 128), $b_2$ is shape $(1, h_2)$, and $Z_2$ is shape $(n, h_2)$.
   - ReLU activation:
     $$
     A_2 = \text{ReLU}(Z_2) = \max(0, Z_2)
     $$

3. **Third Layer (Output Layer)**:
   - Linear combination:
     $$
     Z_3 = A_2 W_3 + b_3
     $$
     Where $W_3$ is shape $(h_2, k)$ ($k$ is number of classes, e.g., 10 for MNIST), $b_3$ is shape $(1, k)$, and $Z_3$ is shape $(n, k)$.
   - Softmax activation:
     $$
     A_3 = \text{softmax}(Z_3)_i = \frac{e^{Z_{3,i}}}{\sum_{j=1}^k e^{Z_{3,j}}}
     $$

### Backpropagation

Backpropagation computes gradients by propagating errors backward. For categorical cross-entropy loss $L = -\frac{1}{n} \sum y \log(A_3)$, the steps are:

1. **Output Layer (Layer 3)**:
   - Error term:
     $$
     \delta_3 = A_3 - y
     $$
     Shape $(n, k)$.
   - Gradients:
     $$
     \nabla W_3 = \frac{1}{n} A_2^T \delta_3, \quad \nabla b_3 = \frac{1}{n} \sum_{i=1}^n \delta_{3,i}
     $$

2. **Second Hidden Layer (Layer 2)**:
   - Error term:
     $$
     \delta_2 = (\delta_3 W_3^T) \cdot \text{ReLU}'(Z_2)
     $$
     Where $\text{ReLU}'(Z_2) = 1$ if $Z_2 > 0$, else 0, shape $(n, h_2)$.
   - Gradients:
     $$
     \nabla W_2 = \frac{1}{n} A_1^T \delta_2, \quad \nabla b_2 = \frac{1}{n} \sum_{i=1}^n \delta_{2,i}
     $$

3. **First Hidden Layer (Layer 1)**:
   - Error term:
     $$
     \delta_1 = (\delta_2 W_2^T) \cdot \text{ReLU}'(Z_1)
     $$
     Shape $(n, h_1)$.
   - Gradients:
     $$
     \nabla W_1 = \frac{1}{n} X^T \delta_1, \quad \nabla b_1 = \frac{1}{n} \sum_{i=1}^n \delta_{1,i}
     $$

### Vanishing Gradients

As we add layers, gradients in earlier layers (e.g., $\delta_1$) are products of multiple terms (e.g., $\delta_3 W_3^T W_2^T \cdot$ derivatives). If these terms are small (e.g., due to activations like sigmoid with derivatives < 1), gradients can vanish, stalling learning. ReLU mitigates this by keeping derivatives at 1 for positive inputs, but deeper networks may still struggle. We’ll observe gradient magnitudes during training to illustrate this challenge.

Now, let’s implement a 3-layer MLP and train it on MNIST.

---

## Implementing a 3-Layer MLP with NumPy

We’ll extend our `forward_mlp()` and `backward_mlp()` functions to support a 3-layer architecture (784 → 256 → 128 → 10 for MNIST). We’ll train the network using mini-batch gradient descent and visualize loss and accuracy over epochs with `matplotlib`.

### Forward Pass for 3-Layer MLP

Here’s the updated forward pass for a 3-layer MLP:

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple

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
```

### Backpropagation for 3-Layer MLP

Here’s the updated backpropagation to compute gradients for all parameters in a 3-layer MLP:

```python
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
```

### Training Loop for 3-Layer MLP on MNIST

Now, let’s train a 3-layer MLP on MNIST using mini-batch gradient descent, plotting loss and accuracy over epochs with `matplotlib`.

**Note**: Ensure you have `sklearn` (`pip install scikit-learn`) for loading MNIST and `matplotlib` (`pip install matplotlib`) for plotting. We’ll use a subset of MNIST for CPU efficiency.

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, List
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_network import normalize, relu, softmax, cross_entropy

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

# Training loop
lr = 0.1
num_epochs = 20
batch_size = 64
n_samples = X_train.shape[0]
loss_history = []
accuracy_history = []

print("Starting training...")
for epoch in range(num_epochs):
    # Shuffle data
    indices = np.random.permutation(n_samples)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    # Mini-batch processing
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        batch_size_actual = X_batch.shape[0]
        
        # Forward pass
        A1, A2, A3 = forward_mlp_3layer(X_batch, W1, b1, W2, b2, W3, b3)
        
        # Compute gradients via backpropagation
        Z1 = X_batch @ W1 + b1
        Z2 = A1 @ W2 + b2
        grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = backward_mlp_3layer(
            X_batch, A1, A2, A3, y_batch, W1, W2, W3, Z1, Z2)
        
        # Update parameters
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        W2 -= lr * grad_W2
        b2 -= lr * grad_b2
        W3 -= lr * grad_W3
        b3 -= lr * grad_b3
    
    # Compute loss on full training set at end of epoch
    _, _, A3_full = forward_mlp_3layer(X_train, W1, b1, W2, b2, W3, b3)
    loss = cross_entropy(A3_full, y_train)
    loss_history.append(loss)
    
    # Compute accuracy on test set
    _, _, A3_test = forward_mlp_3layer(X_test, W1, b1, W2, b2, W3, b3)
    predictions = np.argmax(A3_test, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == true_labels)
    accuracy_history.append(accuracy)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Plot loss and accuracy history
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.plot(range(1, num_epochs + 1), loss_history, label='Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.set_title('Training Loss Over Epochs')
ax1.legend()
ax1.grid(True)

ax2.plot(range(1, num_epochs + 1), accuracy_history, label='Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Test Accuracy Over Epochs')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Final evaluation on test set
_, _, A3_test = forward_mlp_3layer(X_test, W1, b1, W2, b2, W3, b3)
predictions = np.argmax(A3_test, axis=1)
true_labels = np.argmax(y_test, axis=1)
final_accuracy = np.mean(predictions == true_labels)
print("Final Test Accuracy:", final_accuracy)
```

**Output** (approximate, values will vary due to randomness and subset size):
```
Loading MNIST data...
Starting training...
Epoch 1/20, Loss: 2.3021, Test Accuracy: 0.1050
Epoch 2/20, Loss: 2.2903, Test Accuracy: 0.1280
...
Epoch 10/20, Loss: 0.6842, Test Accuracy: 0.8320
...
Epoch 20/20, Loss: 0.3561, Test Accuracy: 0.8870
Final Test Accuracy: 0.8870
```

**Loss and Accuracy Plots**: (Two `matplotlib` plots will display, showing training loss decreasing over epochs from ~2.3 to below 0.5, and test accuracy increasing to ~85-90%.)

In this example, we train a 3-layer MLP on a subset of MNIST (5000 samples for CPU efficiency) with a 784 → 256 → 128 → 10 architecture. Over 20 epochs with a batch size of 64, the loss decreases, and test accuracy reaches ~85-90%, similar to our 2-layer MLP but with potential for better feature extraction due to added depth. Training time is slightly longer (~3-6 minutes on CPU) due to the additional layer, but remains manageable. We’ll discuss if vanishing gradients affect early layer updates by observing training behavior.

---

## Organizing Our Growing Library

Let’s update our `neural_network.py` file to include the `forward_mlp_3layer()` and `backward_mlp_3layer()` functions alongside our previous implementations. These extend our MLP capabilities to deeper architectures.

```python
# neural_network.py
import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable, Tuple, List, Dict

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
```

You can now import these new functions using `from neural_network import forward_mlp_3layer, backward_mlp_3layer`. They extend our MLP capabilities to a deeper 3-layer architecture, allowing us to explore challenges like vanishing gradients.

---

## Exercises: Practice with Deeper MLPs and Vanishing Gradients

To reinforce your understanding of deeper MLPs and the vanishing gradient problem, try these Python-focused coding exercises. They’ll help you build intuition for training deeper networks and observing their behavior. Run the code and compare outputs to verify your solutions.

1. **Forward and Backward Pass on Small Synthetic Data**  
   Create synthetic data `X = np.array([[1.0, 2.0], [3.0, 4.0]])` (2 samples, 2 features) and one-hot labels `y = np.array([[1, 0], [0, 1]])` (2 classes). Initialize a small 3-layer MLP (2 → 4 → 3 → 2) with small random weights (e.g., `W1 = np.random.randn(2, 4) * 0.1`). Compute the forward pass with `forward_mlp_3layer()`, then compute gradients with `backward_mlp_3layer()`. Print the shapes of all activations and gradients to verify correctness.

   ```python
   # Your code here
   X = np.array([[1.0, 2.0], [3.0, 4.0]])
   y = np.array([[1, 0], [0, 1]])
   W1 = np.random.randn(2, 4) * 0.1
   b1 = np.zeros((1, 4))
   W2 = np.random.randn(4, 3) * 0.1
   b2 = np.zeros((1, 3))
   W3 = np.random.randn(3, 2) * 0.1
   b3 = np.zeros((1, 2))
   A1, A2, A3 = forward_mlp_3layer(X, W1, b1, W2, b2, W3, b3)
   Z1 = X @ W1 + b1
   Z2 = A1 @ W2 + b2
   grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = backward_mlp_3layer(
       X, A1, A2, A3, y, W1, W2, W3, Z1, Z2)
   print("A1 shape:", A1.shape)
   print("A2 shape:", A2.shape)
   print("A3 shape:", A3.shape)
   print("Gradient W1 shape:", grad_W1.shape)
   print("Gradient b1 shape:", grad_b1.shape)
   print("Gradient W2 shape:", grad_W2.shape)
   print("Gradient b2 shape:", grad_b2.shape)
   print("Gradient W3 shape:", grad_W3.shape)
   print("Gradient b3 shape:", grad_b3.shape)
   ```

2. **Single Epoch Training on Synthetic Data**  
   Using the data and parameters from Exercise 1, perform one epoch of training: compute forward pass, backpropagation gradients, and update parameters with `lr = 0.1`. Compute loss before and after the update using `cross_entropy()` to see if it decreases. Print initial and final loss.

   ```python
   # Your code here
   X = np.array([[1.0, 2.0], [3.0, 4.0]])
   y = np.array([[1, 0], [0, 1]])
   W1 = np.random.randn(2, 4) * 0.1
   b1 = np.zeros((1, 4))
   W2 = np.random.randn(4, 3) * 0.1
   b2 = np.zeros((1, 3))
   W3 = np.random.randn(3, 2) * 0.1
   b3 = np.zeros((1, 2))
   lr = 0.1
   # Initial forward pass and loss
   A1, A2, A3 = forward_mlp_3layer(X, W1, b1, W2, b2, W3, b3)
   initial_loss = cross_entropy(A3, y)
   # Backpropagation and update
   Z1 = X @ W1 + b1
   Z2 = A1 @ W2 + b2
   grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = backward_mlp_3layer(
       X, A1, A2, A3, y, W1, W2, W3, Z1, Z2)
   W1 -= lr * grad_W1
   b1 -= lr * grad_b1
   W2 -= lr * grad_W2
   b2 -= lr * grad_b2
   W3 -= lr * grad_W3
   b3 -= lr * grad_b3
   # Final forward pass and loss
   A1, A2, A3 = forward_mlp_3layer(X, W1, b1, W2, b2, W3, b3)
   final_loss = cross_entropy(A3, y)
   print("Initial Loss:", initial_loss)
   print("Final Loss after one update:", final_loss)
   ```

3. **Training a 3-Layer MLP on Small MNIST Subset**  
   Load a small subset of MNIST (e.g., 1000 samples) using `fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)`. Initialize a 784 → 256 → 128 → 10 MLP, train for 10 epochs with `lr = 0.1` and `batch_size = 32`. Print test accuracy after training and plot loss over epochs. Observe if adding a third layer improves accuracy compared to a 2-layer MLP.

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
   n_samples = X_train.shape[0]
   loss_history = []
   for epoch in range(num_epochs):
       indices = np.random.permutation(n_samples)
       X_shuffled = X_train[indices]
       y_shuffled = y_train[indices]
       for start_idx in range(0, n_samples, batch_size):
           end_idx = min(start_idx + batch_size, n_samples)
           X_batch = X_shuffled[start_idx:end_idx]
           y_batch = y_shuffled[start_idx:end_idx]
           A1, A2, A3 = forward_mlp_3layer(X_batch, W1, b1, W2, b2, W3, b3)
           Z1 = X_batch @ W1 + b1
           Z2 = A1 @ W2 + b2
           grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = backward_mlp_3layer(
               X_batch, A1, A2, A3, y_batch, W1, W2, W3, Z1, Z2)
           W1 -= lr * grad_W1
           b1 -= lr * grad_b1
           W2 -= lr * grad_W2
           b2 -= lr * grad_b2
           W3 -= lr * grad_W3
           b3 -= lr * grad_b3
       _, _, A3_full = forward_mlp_3layer(X_train, W1, b1, W2, b2, W3, b3)
       loss = cross_entropy(A3_full, y_train)
       loss_history.append(loss)
       print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
   _, _, A3_test = forward_mlp_3layer(X_test, W1, b1, W2, b2, W3, b3)
   predictions = np.argmax(A3_test, axis=1)
   true_labels = np.argmax(y_test, axis=1)
   accuracy = np.mean(predictions == true_labels)
   print("Test Accuracy:", accuracy)
   plt.plot(range(1, num_epochs + 1), loss_history, label='Training Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Cross-Entropy Loss')
   plt.title('Training Loss Over Epochs')
   plt.legend()
   plt.grid(True)
   plt.show()
   ```

4. **Observing Gradient Magnitudes for Vanishing Gradients**  
   Modify Exercise 3 to print the average magnitude (e.g., `np.mean(np.abs(grad_W1))`) of gradients for `W1`, `W2`, and `W3` during the first epoch. Observe if gradients for `W1` (earliest layer) are significantly smaller than for `W3` (last layer), indicating potential vanishing gradient effects. Note any differences in training stability or convergence speed compared to a 2-layer MLP.

   ```python
   # Your code here
   from sklearn.datasets import fetch_openml
   from sklearn.model_selection import train_test_split
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
   n_samples = X_train.shape[0]
   loss_history = []
   for epoch in range(num_epochs):
       indices = np.random.permutation(n_samples)
       X_shuffled = X_train[indices]
       y_shuffled = y_train[indices]
       if epoch == 0:  # Only print gradients for first epoch
           print("Gradient magnitudes in first epoch:")
       for start_idx in range(0, n_samples, batch_size):
           end_idx = min(start_idx + batch_size, n_samples)
           X_batch = X_shuffled[start_idx:end_idx]
           y_batch = y_shuffled[start_idx:end_idx]
           A1, A2, A3 = forward_mlp_3layer(X_batch, W1, b1, W2, b2, W3, b3)
           Z1 = X_batch @ W1 + b1
           Z2 = A1 @ W2 + b2
           grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = backward_mlp_3layer(
               X_batch, A1, A2, A3, y_batch, W1, W2, W3, Z1, Z2)
           if epoch == 0 and start_idx == 0:  # Print for first batch of first epoch
               print(f"Avg |grad_W1| (first layer): {np.mean(np.abs(grad_W1)):.6f}")
               print(f"Avg |grad_W2| (second layer): {np.mean(np.abs(grad_W2)):.6f}")
               print(f"Avg |grad_W3| (third layer): {np.mean(np.abs(grad_W3)):.6f}")
           W1 -= lr * grad_W1
           b1 -= lr * grad_b1
           W2 -= lr * grad_W2
           b2 -= lr * grad_b2
           W3 -= lr * grad_W3
           b3 -= lr * grad_b3
       _, _, A3_full = forward_mlp_3layer(X_train, W1, b1, W2, b2, W3, b3)
       loss = cross_entropy(A3_full, y_train)
       loss_history.append(loss)
       print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
   _, _, A3_test = forward_mlp_3layer(X_test, W1, b1, W2, b2, W3, b3)
   predictions = np.argmax(A3_test, axis=1)
   true_labels = np.argmax(y_test, axis=1)
   accuracy = np.mean(predictions == true_labels)
   print("Test Accuracy:", accuracy)
   ```

These exercises will help you build intuition for implementing and training deeper MLPs, extending forward and backward passes to additional layers, and observing challenges like vanishing gradients through gradient magnitudes.

---

## Closing Thoughts

Congratulations on implementing a 3-layer MLP and exploring the vanishing gradient challenge! In this post, we’ve extended our neural network to a deeper architecture with `forward_mlp_3layer()` and `backward_mlp_3layer()`, trained it on MNIST achieving ~85-90% accuracy, and visualized loss and accuracy over epochs. We’ve also discussed how vanishing gradients can hinder training in early layers of deeper networks, a key issue in deep learning.

In the next chapter (Part 4.2: _Initialization and Normalization Techniques_), we’ll address vanishing gradients and other training challenges with solutions like better weight initialization (e.g., Xavier) and batch normalization, improving stability and performance for deeper networks.

Until then, experiment with the code and exercises above. If you have questions or want to share your solutions, drop a comment below—I’m excited to hear from you. Let’s keep building our deep learning toolkit together!

**Next Up**: Part 4.2 – Initialization and Normalization Techniques
