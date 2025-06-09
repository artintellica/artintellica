+++
title = "Learn Deep Learning with NumPy, Part 3.4: Backpropagation for Training MLPs"
author = "Artintellica"
date = "2025-06-09"
+++

## Introduction

Welcome back to our blog series, _"Learn Deep Learning with NumPy"_! In Part
3.3, we implemented forward propagation for a 2-layer Multi-Layer Perceptron
(MLP), transforming input data through hidden and output layers to produce
predictions for MNIST digit classification. Now, in Part 3.4, we’ll complete the
training process by implementing _backpropagation_—the algorithm that computes
gradients of the loss with respect to each parameter, enabling us to update
weights and biases using gradient descent.

By the end of this post, you’ll understand the mathematics of backpropagation,
implement it in NumPy with `backward_mlp()` to compute gradients, and train a
2-layer MLP on MNIST, achieving ~85-90% accuracy. We’ll also visualize the
training progress using loss plots with `matplotlib`. This marks the completion
of our first full neural network implementation. Let’s dive into the math and
code for training MLPs!

---

## Why Backpropagation Matters in Deep Learning

Forward propagation computes predictions, but to train a neural network, we need
to adjust parameters (weights and biases) to minimize the loss function.
_Backpropagation_ (short for "backward propagation of errors") is the
cornerstone of neural network training. It uses the chain rule from calculus to
compute gradients of the loss with respect to each parameter by propagating
errors backward through the network. These gradients tell us how to update
parameters via gradient descent to reduce the loss.

In deep learning, backpropagation is essential because:

- It efficiently calculates gradients for all parameters, even in deep networks
  with many layers.
- It enables learning by iteratively improving predictions through parameter
  updates.
- It scales to complex architectures, making modern deep learning possible.

In this post, we’ll implement backpropagation for our 2-layer MLP, compute
gradients for weights and biases, and train the network on MNIST using
mini-batch gradient descent. Let’s explore the math behind this powerful
algorithm.

---

## Mathematical Foundations: Backpropagation in a 2-Layer MLP

Backpropagation computes gradients of the loss function with respect to each
parameter by applying the chain rule layer by layer, starting from the output
and moving backward to the input. For our 2-layer MLP with forward pass
$X \rightarrow Z_1 = XW_1 + b_1 \rightarrow A_1 = \text{ReLU}(Z_1) \rightarrow Z_2 = A_1 W_2 + b_2 \rightarrow A_2 = \text{softmax}(Z_2)$,
and categorical cross-entropy loss $L = -\frac{1}{n} \sum y \log(A_2)$, the
steps are as follows for a batch of $n$ samples.

### Output Layer Gradients (Layer 2)

1. **Error at Output (Delta for Layer 2)**: Since we use softmax with
   cross-entropy loss, the error term (or delta) for the output layer simplifies
   nicely due to the properties of softmax and cross-entropy derivatives:

   $$
   \delta_2 = A_2 - y
   $$

   Where $A_2$ is the softmax output (shape $n \times k$, $k$ classes), and $y$
   is the one-hot encoded true labels (shape $n \times k$). This is the
   difference between predicted probabilities and true labels.

2. **Gradients for Output Layer Parameters**:
   - Gradient for weights $W_2$ (shape $h \times k$, where $h$ is hidden layer
     size):
     $$
     \nabla W_2 = \frac{1}{n} A_1^T \delta_2
     $$
     Where $A_1$ is the hidden layer output (shape $n \times h$).
   - Gradient for bias $b_2$ (shape $1 \times k$):
     $$
     \nabla b_2 = \frac{1}{n} \sum_{i=1}^n \delta_{2,i}
     $$
     (Mean of $\delta_2$ across samples, computed row-wise.)

### Hidden Layer Gradients (Layer 1)

3. **Error at Hidden Layer (Delta for Layer 1)**: Propagate the error backward
   from layer 2 to layer 1 using the chain rule:

   $$
   \delta_1 = (\delta_2 W_2^T) \cdot \text{ReLU}'(Z_1)
   $$

   Where:

   - $\delta_2 W_2^T$ (shape $n \times h$) propagates the error through $W_2$
     (shape $h \times k$).
   - $\text{ReLU}'(Z_1)$ is the derivative of ReLU at $Z_1$ (shape
     $n \times h$), which is 1 where $Z_1 > 0$ and 0 otherwise, applied
     element-wise with the dot product $\cdot$.
   - $\delta_1$ has shape $n \times h$.

4. **Gradients for Hidden Layer Parameters**:
   - Gradient for weights $W_1$ (shape $d \times h$, where $d$ is input
     features):
     $$
     \nabla W_1 = \frac{1}{n} X^T \delta_1
     $$
     Where $X$ is the input (shape $n \times d$).
   - Gradient for bias $b_1$ (shape $1 \times h$):
     $$
     \nabla b_1 = \frac{1}{n} \sum_{i=1}^n \delta_{1,i}
     $$
     (Mean of $\delta_1$ across samples, row-wise.)

### Parameter Updates

Using these gradients, update parameters with gradient descent:

$$
W_1 \leftarrow W_1 - \eta \nabla W_1, \quad b_1 \leftarrow b_1 - \eta \nabla b_1, \quad W_2 \leftarrow W_2 - \eta \nabla W_2, \quad b_2 \leftarrow b_2 - \eta \nabla b_2
$$

Where $\eta$ is the learning rate. This process repeats for each mini-batch over
multiple epochs until the loss converges or a set number of iterations is
reached. Now, let’s implement backpropagation and train our MLP on MNIST.

---

## Implementing Backpropagation and Training with NumPy

We’ll create a `backward_mlp()` function to compute gradients for our 2-layer
MLP and integrate it with a training loop using mini-batch gradient descent.
We’ll train on the MNIST dataset (10-class digit classification) and visualize
the loss over epochs with `matplotlib`.

### Backpropagation for 2-Layer MLP

Here’s the implementation of backpropagation to compute gradients for all
parameters:

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, Dict

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
```

### Training Loop for MLP

Now, let’s create a training loop to apply forward propagation, compute
gradients with backpropagation, and update parameters using gradient descent.
We’ll train on MNIST and plot the loss over epochs.

**Note**: Ensure you have `sklearn` (`pip install scikit-learn`) for loading
MNIST and `matplotlib` (`pip install matplotlib`) for plotting. We’ll use a
subset of MNIST for CPU efficiency.

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, List
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from neural_network import normalize, relu, softmax, cross_entropy, forward_mlp

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

# Initialize parameters for 2-layer MLP (784 -> 256 -> 10)
n_features = X_train.shape[1]  # 784 for MNIST
n_hidden = 256
W1 = np.random.randn(n_features, n_hidden) * 0.01
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_classes) * 0.01
b2 = np.zeros((1, n_classes))

# Training loop
lr = 0.1
num_epochs = 20
batch_size = 64
n_samples = X_train.shape[0]
loss_history = []

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
        A1, A2 = forward_mlp(X_batch, W1, b1, W2, b2)

        # Compute gradients via backpropagation
        Z1 = X_batch @ W1 + b1  # Pre-activation for ReLU
        grad_W1, grad_b1, grad_W2, grad_b2 = backward_mlp(X_batch, A1, A2, y_batch, W1, W2, Z1)

        # Update parameters
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        W2 -= lr * grad_W2
        b2 -= lr * grad_b2

    # Compute loss on full training set at end of epoch
    _, A2_full = forward_mlp(X_train, W1, b1, W2, b2)
    loss = cross_entropy(A2_full, y_train)
    loss_history.append(loss)

    # Compute accuracy on test set
    _, A2_test = forward_mlp(X_test, W1, b1, W2, b2)
    predictions = np.argmax(A2_test, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == true_labels)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Plot loss history
plt.plot(range(1, num_epochs + 1), loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Final evaluation on test set
_, A2_test = forward_mlp(X_test, W1, b1, W2, b2)
predictions = np.argmax(A2_test, axis=1)
true_labels = np.argmax(y_test, axis=1)
final_accuracy = np.mean(predictions == true_labels)
print("Final Test Accuracy:", final_accuracy)
```

**Output** (approximate, values will vary due to randomness and subset size):

```
Loading MNIST data...
Starting training...
Epoch 1/20, Loss: 2.3010, Test Accuracy: 0.1120
Epoch 2/20, Loss: 2.2895, Test Accuracy: 0.1340
...
Epoch 10/20, Loss: 0.6234, Test Accuracy: 0.8450
...
Epoch 20/20, Loss: 0.3127, Test Accuracy: 0.8920
Final Test Accuracy: 0.8920
```

**Loss Plot**: (A matplotlib plot will display, showing the training loss
decreasing over epochs, typically from ~2.3 to below 0.5.)

In this example, we train a 2-layer MLP on a subset of MNIST (5000 samples for
CPU efficiency) with a 784 → 256 → 10 architecture. Over 20 epochs with a batch
size of 64, the loss decreases, and test accuracy reaches ~85-90%, demonstrating
effective learning. The loss plot visualizes training progress, confirming
convergence. Training time is manageable on a CPU (~2-5 minutes total) due to
the subset and mini-batch approach.

---

## Organizing Our Growing Library

Let’s update our `neural_network.py` file to include the `backward_mlp()`
function alongside our previous implementations. This will complete the core
components needed for training a 2-layer MLP.

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
```

You can now import this new function using
`from neural_network import backward_mlp`. Combined with `forward_mlp()`, it
completes the core functionality for training a 2-layer MLP on multi-class
classification tasks like MNIST.

---

## Exercises: Practice with Backpropagation and MLP Training

To reinforce your understanding of backpropagation and training MLPs, try these
Python-focused coding exercises. They’ll solidify your grasp of gradient
computation and parameter updates. Run the code and compare outputs to verify
your solutions.

1. **Backpropagation on Small Synthetic Data**  
   Create synthetic data `X = np.array([[1.0, 2.0], [3.0, 4.0]])` (2 samples, 2
   features) and one-hot labels `y = np.array([[1, 0], [0, 1]])` (2 classes).
   Initialize a small 2-layer MLP (2 → 3 → 2) with
   `W1 = np.random.randn(2, 3) * 0.1`, `b1 = np.zeros((1, 3))`,
   `W2 = np.random.randn(3, 2) * 0.1`, `b2 = np.zeros((1, 2))`. Compute the
   forward pass with `forward_mlp()`, then compute gradients with
   `backward_mlp()`. Print the shapes of all gradients to verify correctness.

   ```python
   # Your code here
   X = np.array([[1.0, 2.0], [3.0, 4.0]])
   y = np.array([[1, 0], [0, 1]])
   W1 = np.random.randn(2, 3) * 0.1
   b1 = np.zeros((1, 3))
   W2 = np.random.randn(3, 2) * 0.1
   b2 = np.zeros((1, 2))
   A1, A2 = forward_mlp(X, W1, b1, W2, b2)
   Z1 = X @ W1 + b1  # Pre-activation for hidden layer
   grad_W1, grad_b1, grad_W2, grad_b2 = backward_mlp(X, A1, A2, y, W1, W2, Z1)
   print("Gradient W1 shape:", grad_W1.shape)
   print("Gradient b1 shape:", grad_b1.shape)
   print("Gradient W2 shape:", grad_W2.shape)
   print("Gradient b2 shape:", grad_b2.shape)
   ```

2. **Single Epoch Training on Synthetic Data**  
   Using the data and parameters from Exercise 1, perform one epoch of training:
   compute forward pass, backpropagation gradients, and update parameters with
   `lr = 0.1`. Compute loss before and after the update using `cross_entropy()`
   to see if it decreases. Print initial and final loss.

   ```python
   # Your code here
   X = np.array([[1.0, 2.0], [3.0, 4.0]])
   y = np.array([[1, 0], [0, 1]])
   W1 = np.random.randn(2, 3) * 0.1
   b1 = np.zeros((1, 3))
   W2 = np.random.randn(3, 2) * 0.1
   b2 = np.zeros((1, 2))
   lr = 0.1
   # Initial forward pass and loss
   A1, A2 = forward_mlp(X, W1, b1, W2, b2)
   initial_loss = cross_entropy(A2, y)
   # Backpropagation and update
   Z1 = X @ W1 + b1
   grad_W1, grad_b1, grad_W2, grad_b2 = backward_mlp(X, A1, A2, y, W1, W2, Z1)
   W1 -= lr * grad_W1
   b1 -= lr * grad_b1
   W2 -= lr * grad_W2
   b2 -= lr * grad_b2
   # Final forward pass and loss
   A1, A2 = forward_mlp(X, W1, b1, W2, b2)
   final_loss = cross_entropy(A2, y)
   print("Initial Loss:", initial_loss)
   print("Final Loss after one update:", final_loss)
   ```

3. **Mini-Batch Training on Small MNIST Subset**  
   Load a small subset of MNIST (e.g., 500 samples) using
   `fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)`.
   Limit to digits 0 and 1 for binary classification (convert labels to
   one-hot). Initialize a 784 → 128 → 2 MLP, train for 10 epochs with `lr = 0.1`
   and `batch_size = 32`. Print test accuracy after training. Note: Use a binary
   version for simplicity, adapting `softmax` to `sigmoid` and `cross_entropy`
   to `binary_cross_entropy` if needed.

   ```python
   # Your code here
   from sklearn.datasets import fetch_openml
   from sklearn.model_selection import train_test_split
   X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
   X_full = X_full.astype(float)
   y_full = y_full.astype(int)
   mask = (y_full == 0) | (y_full == 1)
   X = X_full[mask][:500]
   y = y_full[mask][:500]
   y_one_hot = np.zeros((y.shape[0], 2))
   y_one_hot[y == 0, 0] = 1
   y_one_hot[y == 1, 1] = 1
   X = normalize(X)
   X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
   W1 = np.random.randn(784, 128) * 0.01
   b1 = np.zeros((1, 128))
   W2 = np.random.randn(128, 2) * 0.01
   b2 = np.zeros((1, 2))
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
           A1, A2 = forward_mlp(X_batch, W1, b1, W2, b2)
           Z1 = X_batch @ W1 + b1
           grad_W1, grad_b1, grad_W2, grad_b2 = backward_mlp(X_batch, A1, A2, y_batch, W1, W2, Z1)
           W1 -= lr * grad_W1
           b1 -= lr * grad_b1
           W2 -= lr * grad_W2
           b2 -= lr * grad_b2
       _, A2_full = forward_mlp(X_train, W1, b1, W2, b2)
       loss = cross_entropy(A2_full, y_train)
       loss_history.append(loss)
       print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
   _, A2_test = forward_mlp(X_test, W1, b1, W2, b2)
   predictions = np.argmax(A2_test, axis=1)
   true_labels = np.argmax(y_test, axis=1)
   accuracy = np.mean(predictions == true_labels)
   print("Test Accuracy:", accuracy)
   ```

4. **Impact of Learning Rate on Training**  
   Using the setup from Exercise 3, train the same MLP with a higher learning
   rate `lr = 1.0` for 10 epochs. Compare the loss history and final test
   accuracy to Exercise 3. Observe if the higher learning rate leads to faster
   convergence, instability, or divergence.

   ```python
   # Your code here
   from sklearn.datasets import fetch_openml
   from sklearn.model_selection import train_test_split
   X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
   X_full = X_full.astype(float)
   y_full = y_full.astype(int)
   mask = (y_full == 0) | (y_full == 1)
   X = X_full[mask][:500]
   y = y_full[mask][:500]
   y_one_hot = np.zeros((y.shape[0], 2))
   y_one_hot[y == 0, 0] = 1
   y_one_hot[y == 1, 1] = 1
   X = normalize(X)
   X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
   W1 = np.random.randn(784, 128) * 0.01
   b1 = np.zeros((1, 128))
   W2 = np.random.randn(128, 2) * 0.01
   b2 = np.zeros((1, 2))
   lr = 1.0  # Higher learning rate
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
           A1, A2 = forward_mlp(X_batch, W1, b1, W2, b2)
           Z1 = X_batch @ W1 + b1
           grad_W1, grad_b1, grad_W2, grad_b2 = backward_mlp(X_batch, A1, A2, y_batch, W1, W2, Z1)
           W1 -= lr * grad_W1
           b1 -= lr * grad_b1
           W2 -= lr * grad_W2
           b2 -= lr * grad_b2
       _, A2_full = forward_mlp(X_train, W1, b1, W2, b2)
       loss = cross_entropy(A2_full, y_train)
       loss_history.append(loss)
       print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
   _, A2_test = forward_mlp(X_test, W1, b1, W2, b2)
   predictions = np.argmax(A2_test, axis=1)
   true_labels = np.argmax(y_test, axis=1)
   accuracy = np.mean(predictions == true_labels)
   print("Test Accuracy:", accuracy)
   ```

These exercises will help you build intuition for backpropagation, gradient
computation in MLPs, and the impact of hyperparameters like learning rate on
training performance.

---

## Closing Thoughts

Congratulations on completing your first full neural network implementation with
backpropagation! In this post, we’ve explored the mathematics of
backpropagation, implemented `backward_mlp()` to compute gradients, and trained
a 2-layer MLP on MNIST, achieving ~85-90% test accuracy. We’ve also visualized
training progress with loss plots using `matplotlib`, confirming effective
learning.

With Module 3 nearly complete, we’ve built a solid foundation for neural
networks, from single-layer perceptrons to multi-layer models with forward and
backward passes. In the next chapter (Part 4.1: _Deeper MLPs and Vanishing
Gradients_), we’ll extend our MLP to deeper architectures, exploring challenges
like vanishing gradients and strategies to address them.

Until then, experiment with the code and exercises above. If you have questions
or want to share your solutions, drop a comment below—I’m excited to hear from
you. Let’s keep building our deep learning toolkit together!

**Next Up**: Part 4.1 – Deeper MLPs and Vanishing Gradients
