+++
title = "Learn Deep Learning with NumPy, Part 3.3: Multi-Layer Perceptrons and Forward Propagation"
author = "Artintellica"
date = "2025-06-05"
+++

# Learn Deep Learning with NumPy, Part 3.3: Multi-Layer Perceptrons and Forward Propagation

## Introduction

Welcome back to our blog series, _"Learn Deep Learning with NumPy"_! In Part 3.2, we expanded our toolkit with activation functions like ReLU and softmax, crucial for introducing non-linearity into neural networks. Now, in Part 3.3, we’re taking a significant step forward by implementing a *Multi-Layer Perceptron (MLP)*—a neural network with multiple layers. We’ll focus on the forward propagation step, which computes predictions through a series of transformations, and apply it to the MNIST dataset for digit classification.

By the end of this post, you’ll understand how a 2-layer MLP processes input data through hidden and output layers, implement forward propagation in NumPy with `forward_mlp()`, and compute multi-class cross-entropy loss. This builds on our previous functions like `relu()` and `softmax()`, setting the stage for full training with backpropagation in Part 3.4. Let’s dive into the math and code for multi-layer neural networks!

---

## Why Multi-Layer Perceptrons Matter in Deep Learning

A single-layer perceptron, as we saw in Part 3.1, is limited to solving linearly separable problems. It struggles with tasks like XOR or complex datasets like MNIST because it can only draw a straight decision boundary. *Multi-Layer Perceptrons (MLPs)* overcome this by stacking multiple layers of neurons, each with non-linear activation functions. This allows the network to learn hierarchical features and model non-linear relationships, making it capable of tackling intricate patterns in data.

In deep learning, MLPs are foundational:
- **Hidden Layers**: Introduce non-linearity (via activations like ReLU) to learn complex mappings.
- **Output Layer**: Produces final predictions, often with softmax for multi-class classification.
- **Forward Propagation**: Computes predictions by passing input through layers, which is the first step before training with backpropagation.

In this post, we’ll implement a 2-layer MLP for MNIST digit classification (10 classes), focusing on forward propagation. Our network will transform 784-pixel images into a 256-neuron hidden layer with ReLU, then into a 10-neuron output layer with softmax. Let’s explore the math behind this process.

---

## Mathematical Foundations: Forward Propagation in a 2-Layer MLP

Forward propagation in a 2-layer MLP processes input data through two successive transformations to produce predictions. For an input matrix $X$ of shape $(n, d)$ (where $n$ is the number of samples and $d$ is the number of features, e.g., 784 for MNIST), the steps are:

1. **First Layer (Hidden Layer)**:
   - Compute linear combination:
     $$
     Z_1 = X W_1 + b_1
     $$
     Where $W_1$ is the weight matrix of shape $(d, h)$ ($h$ is hidden layer size, e.g., 256), and $b_1$ is the bias of shape $(1, h)$, broadcasted to match $Z_1$’s shape $(n, h)$.
   - Apply ReLU activation:
     $$
     A_1 = \text{ReLU}(Z_1) = \max(0, Z_1)
     $$
     Applied element-wise, introducing non-linearity.

2. **Second Layer (Output Layer)**:
   - Compute linear combination:
     $$
     Z_2 = A_1 W_2 + b_2
     $$
     Where $W_2$ is the weight matrix of shape $(h, k)$ ($k$ is number of classes, e.g., 10 for MNIST), and $b_2$ is the bias of shape $(1, k)$, resulting in $Z_2$ of shape $(n, k)$.
   - Apply softmax activation for multi-class probabilities:
     $$
     A_2 = \text{softmax}(Z_2)_i = \frac{e^{Z_{2,i}}}{\sum_{j=1}^k e^{Z_{2,j}}}
     $$
     Applied row-wise, ensuring each sample’s outputs sum to 1.

3. **Loss Function (Cross-Entropy)**:
   For multi-class classification, we use categorical cross-entropy loss to measure error between predictions $A_2$ (probabilities) and true labels $y$ (one-hot encoded, shape $(n, k)$):
   $$
   L = -\frac{1}{n} \sum_{i=1}^n \sum_{j=1}^k y_{i,j} \log(A_{2,i,j})
   $$
   This loss penalizes incorrect class predictions more heavily when the predicted probability for the true class is low.

This forward pass—layer-by-layer transformation with activations—defines how an MLP computes predictions. Gradients for training (via backpropagation) will be covered in Part 3.4. Now, let’s implement forward propagation for a 2-layer MLP in NumPy.

---

## Implementing a 2-Layer MLP Forward Pass with NumPy

We’ll create a reusable `forward_mlp()` function for a 2-layer MLP, using `relu()` for the hidden layer and `softmax()` for the output layer. We’ll also implement a `cross_entropy()` loss function for multi-class classification. Our example will simulate MNIST data processing with a network of size 784 (input) → 256 (hidden, ReLU) → 10 (output, softmax).

### Forward Pass for 2-Layer MLP

Here’s the implementation of forward propagation for a 2-layer MLP:

```python
import numpy as np
from numpy.typing import NDArray
from typing import Union

def forward_mlp(X: NDArray[np.floating], W1: NDArray[np.floating], b1: NDArray[np.floating], 
                W2: NDArray[np.floating], b2: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
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
```

### Cross-Entropy Loss for Multi-Class Classification

Here’s the implementation of categorical cross-entropy loss, used to evaluate the MLP’s predictions:

```python
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
```

### Example: Forward Pass for MNIST Classification

Let’s test the forward pass of our 2-layer MLP on synthetic data simulating MNIST inputs (784 features) with a network structure of 784 → 256 (hidden, ReLU) → 10 (output, softmax). We’ll also compute the cross-entropy loss against one-hot encoded labels.

```python
# Simulate MNIST data (4 samples for simplicity)
n_samples = 4
n_features = 784  # MNIST image size (28x28)
n_hidden = 256    # Hidden layer size
n_classes = 10    # MNIST digits (0-9)

# Random input data (simulating normalized MNIST images)
X = np.random.randn(n_samples, n_features)

# Initialize parameters with small random values
W1 = np.random.randn(n_features, n_hidden) * 0.01  # Shape (784, 256)
b1 = np.zeros((1, n_hidden))                       # Shape (1, 256)
W2 = np.random.randn(n_hidden, n_classes) * 0.01   # Shape (256, 10)
b2 = np.zeros((1, n_classes))                      # Shape (1, 10)

# Compute forward pass
A1, A2 = forward_mlp(X, W1, b1, W2, b2)

# Simulate one-hot encoded labels for 4 samples (e.g., digits 3, 7, 1, 9)
y = np.zeros((n_samples, n_classes))
y[0, 3] = 1  # Sample 1: digit 3
y[1, 7] = 1  # Sample 2: digit 7
y[2, 1] = 1  # Sample 3: digit 1
y[3, 9] = 1  # Sample 4: digit 9

# Compute cross-entropy loss
loss = cross_entropy(A2, y)

print("Hidden Layer Output A1 shape (after ReLU):", A1.shape)
print("Output Layer Output A2 shape (after softmax):", A2.shape)
print("Output Probabilities A2 (first few columns):\n", A2[:, :3])
print("Sum of probabilities per sample (should be ~1):\n", np.sum(A2, axis=1))
print("True Labels y (one-hot, first few columns):\n", y[:, :3])
print("Cross-Entropy Loss:", loss)
```

**Output** (approximate, values will vary due to randomness):
```
Hidden Layer Output A1 shape (after ReLU): (4, 256)
Output Layer Output A2 shape (after softmax): (4, 10)
Output Probabilities A2 (first few columns):
 [[0.099 0.101 0.098]
  [0.102 0.097 0.100]
  [0.098 0.103 0.099]
  [0.101 0.096 0.102]]
Sum of probabilities per sample (should be ~1):
 [1. 1. 1. 1.]
True Labels y (one-hot, first few columns):
 [[0. 0. 0.]
  [0. 0. 0.]
  [0. 1. 0.]
  [0. 0. 0.]]
Cross-Entropy Loss: 2.3025
```

In this example, we simulate a forward pass through a 2-layer MLP for MNIST-like data. The hidden layer output `A1` (after ReLU) has shape (4, 256), showing non-linear transformation of the input. The output layer `A2` (after softmax) has shape (4, 10), with each row summing to 1, representing probabilities across 10 digit classes. The cross-entropy loss (~2.3025, close to `ln(10) ≈ 2.3026`) reflects near-random predictions since parameters are initialized randomly, not trained.

---

## Organizing Our Growing Library

Let’s update our `neural_network.py` file to include the `forward_mlp()` and `cross_entropy()` functions alongside our previous implementations. These will be critical for building and evaluating multi-layer neural networks.

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
```

You can now import these new functions using `from neural_network import forward_mlp, cross_entropy`. They will be essential for building and evaluating multi-layer neural networks in the upcoming posts.

---

## Exercises: Practice with Multi-Layer Perceptrons and Forward Propagation

To reinforce your understanding of forward propagation in MLPs, try these Python-focused coding exercises. They’ll prepare you for full training with backpropagation in the next chapter. Run the code and compare outputs to verify your solutions.

1. **Forward Pass on Small Synthetic Data**  
   Create a small synthetic dataset with `X = np.random.randn(2, 3)` (2 samples, 3 features). Initialize parameters for a 2-layer MLP with `W1 = np.random.randn(3, 4) * 0.01` (hidden layer size 4), `b1 = np.zeros((1, 4))`, `W2 = np.random.randn(4, 2) * 0.01` (output layer size 2), and `b2 = np.zeros((1, 2))`. Compute the forward pass using `forward_mlp()` and print the shapes and values of `A1` (hidden layer output) and `A2` (output layer probabilities). Verify that `A2` sums to 1 per sample.

   ```python
   # Your code here
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
   ```

2. **Cross-Entropy Loss on Synthetic Predictions**  
   Using the `A2` from Exercise 1 (output probabilities), create one-hot encoded labels `y = np.array([[1, 0], [0, 1]])` (true classes for 2 samples). Compute the cross-entropy loss using `cross_entropy(A2, y)` and print the result. Verify that the loss is reasonable (e.g., near `ln(2) ≈ 0.693` if predictions are near 0.5).

   ```python
   # Your code here
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
   ```

3. **Forward Pass with MNIST-Like Dimensions**  
   Simulate a small MNIST batch with `X = np.random.randn(5, 784)` (5 samples). Initialize parameters for a 784 → 128 → 10 MLP: `W1 = np.random.randn(784, 128) * 0.01`, `b1 = np.zeros((1, 128))`, `W2 = np.random.randn(128, 10) * 0.01`, `b2 = np.zeros((1, 10))`. Compute the forward pass and print shapes of `A1` and `A2`. Verify `A2` sums to 1 per sample.

   ```python
   # Your code here
   X = np.random.randn(5, 784)
   W1 = np.random.randn(784, 128) * 0.01
   b1 = np.zeros((1, 128))
   W2 = np.random.randn(128, 10) * 0.01
   b2 = np.zeros((1, 10))
   A1, A2 = forward_mlp(X, W1, b1, W2, b2)
   sums = np.sum(A2, axis=1)
   print("Hidden Layer Output A1 shape:", A1.shape)
   print("Output Layer Output A2 shape:", A2.shape)
   print("Sum of probabilities per sample (should be ~1):\n", sums)
   ```

4. **Effect of Parameter Initialization on Outputs**  
   Using the setup from Exercise 3, initialize `W1` and `W2` with larger values (e.g., `np.random.randn(784, 128) * 1.0` and `np.random.randn(128, 10) * 1.0`). Compute the forward pass and compare the distribution of `A2` probabilities to Exercise 3 (e.g., check if one class dominates due to larger weights). Observe how initialization affects softmax outputs.

   ```python
   # Your code here
   X = np.random.randn(5, 784)
   W1 = np.random.randn(784, 128) * 1.0  # Larger initialization
   b1 = np.zeros((1, 128))
   W2 = np.random.randn(128, 10) * 1.0   # Larger initialization
   b2 = np.zeros((1, 10))
   A1, A2 = forward_mlp(X, W1, b1, W2, b2)
   sums = np.sum(A2, axis=1)
   print("Hidden Layer Output A1 shape:", A1.shape)
   print("Output Layer Output A2 shape:", A2.shape)
   print("Output Probabilities A2 (first few columns):\n", A2[:, :3])
   print("Sum of probabilities per sample (should be ~1):\n", sums)
   ```

These exercises will help you build intuition for how forward propagation works in a 2-layer MLP, the role of layer sizes and activations, and the impact of parameter initialization on outputs.

---

## Closing Thoughts

Congratulations on implementing forward propagation for a 2-layer Multi-Layer Perceptron! In this post, we’ve explored how MLPs process data through hidden and output layers with non-linear activations, built `forward_mlp()` reusing `relu()` and `softmax()`, and computed multi-class cross-entropy loss with `cross_entropy()`. This marks a major step toward full neural network training.

In the next chapter (Part 3.4: _Backpropagation for Training MLPs_), we’ll implement backpropagation to compute gradients and train our MLP on MNIST, achieving meaningful digit classification accuracy. This will complete our first full neural network implementation.

Until then, experiment with the code and exercises above. If you have questions or want to share your solutions, drop a comment below—I’m excited to hear from you. Let’s keep building our deep learning toolkit together!

**Next Up**: Part 3.4 – Backpropagation for Training MLPs
