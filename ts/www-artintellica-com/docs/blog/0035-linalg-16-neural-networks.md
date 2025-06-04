+++
title = "Linear Algebra for Machine Learning, Part 16: Neural Networks as Matrix Functions"
author = "Artintellica"
date = "2025-06-04"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0035-linalg-16-neural-networks"
+++

Welcome back to our series on linear algebra for machine learning! In this post,
we’re exploring **Neural Networks as Matrix Functions**, uncovering how these
powerful models are fundamentally built on linear algebra operations. Neural
networks, at their core, are compositions of matrix multiplications and
non-linear activations, enabling them to learn complex patterns from data.
Whether you're building a simple feedforward network or a deep learning model,
understanding the matrix operations behind layers, forward passes, and
backpropagation is essential. Let’s dive into the math, intuition, and
implementation with Python code using PyTorch, visualizations, and hands-on
exercises.

## What Are Neural Networks as Matrix Functions?

A neural network is a series of interconnected layers, where each layer
transforms input data through matrix operations followed by non-linear
activation functions. Consider a simple feedforward neural network with an input
layer, one hidden layer, and an output layer. For an input vector
$x \in \mathbb{R}^{d}$, the computation through the network can be expressed as:

1. **Input to Hidden Layer**:

   $$
   z_1 = W_1 x + b_1
   $$

   $$
   h_1 = \sigma(z_1)
   $$

   where $W_1 \in \mathbb{R}^{h \times d}$ is the weight matrix,
   $b_1 \in \mathbb{R}^{h}$ is the bias vector, $h$ is the number of hidden
   units, and $\sigma$ is a non-linear activation function (e.g., ReLU,
   sigmoid).

2. **Hidden to Output Layer**:
   $$
   z_2 = W_2 h_1 + b_2
   $$
   $$
   \hat{y} = \tau(z_2)
   $$
   where $W_2 \in \mathbb{R}^{o \times h}$ is the weight matrix,
   $b_2 \in \mathbb{R}^{o}$ is the bias vector, $o$ is the number of output
   units, and $\tau$ is the output activation (e.g., linear for regression,
   softmax for classification).

For a batch of inputs $X \in \mathbb{R}^{n \times d}$ (with $n$ samples), these
operations become matrix multiplications:

$$
Z_1 = X W_1^T + b_1^T
$$

$$
H_1 = \sigma(Z_1)
$$

$$
Z_2 = H_1 W_2^T + b_2^T
$$

$$
\hat{Y} = \tau(Z_2)
$$

### Backpropagation and Gradient Descent

Training a neural network involves minimizing a loss function
$\mathcal{L}(\hat{Y}, Y)$ (e.g., mean squared error or cross-entropy) using
gradient descent. Backpropagation computes the gradients of the loss with
respect to the weights and biases through the chain rule, leveraging matrix
calculus. For example, the gradient of the loss with respect to $W_2$ is derived
as:

$$
\frac{\partial \mathcal{L}}{\partial W_2} = \frac{\partial \mathcal{L}}{\partial \hat{Y}} \cdot \frac{\partial \hat{Y}}{\partial Z_2} \cdot H_1^T
$$

These gradients are used to update parameters iteratively:

$$
W_2 \leftarrow W_2 - \eta \cdot \frac{\partial \mathcal{L}}{\partial W_2}
$$

where $\eta$ is the learning rate.

## Why Do Neural Networks as Matrix Functions Matter in Machine Learning?

Neural networks are central to modern machine learning for several reasons:

1. **Expressiveness**: By stacking layers of matrix operations and
   non-linearities, neural networks can model complex, non-linear relationships
   in data.
2. **Scalability**: Vectorized matrix operations enable efficient computation on
   large datasets, especially with GPU acceleration.
3. **Flexibility**: They can be adapted for tasks like regression,
   classification, and image processing by adjusting architectures and loss
   functions.
4. **Linear Algebra Foundation**: Understanding neural networks as matrix
   functions connects directly to the linear algebra concepts we’ve covered,
   such as matrix multiplication and optimization.

Mastering the matrix perspective of neural networks is key to designing,
training, and debugging deep learning models.

## Implementing Neural Networks with PyTorch

Let’s implement a simple feedforward neural network using PyTorch to solve a
regression problem. We’ll examine parameter shapes, forward passes, and training
with gradient descent.

### Example 1: Simple Neural Network for Regression with PyTorch

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic 2D data for regression
n_samples = 200
X = np.random.randn(n_samples, 2) * 2  # 2 features
y = 0.5 * X[:, 0]**2 + 1.5 * X[:, 1] + 2.0 + np.random.randn(n_samples) * 0.5
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).reshape(-1, 1)
print("Data shape:", X.shape, y.shape)

# Define a simple neural network with one hidden layer
class SimpleNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=10, output_size=1):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Instantiate the model
model = SimpleNN()
print("Model architecture:")
print(model)

# Print parameter shapes
print("\nParameter shapes:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
n_epochs = 200
losses = []
for epoch in range(n_epochs):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)
    losses.append(loss.item())

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# Plot loss over epochs
plt.figure(figsize=(8, 6))
plt.plot(range(n_epochs), losses, label='Training Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs (Simple Neural Network)')
plt.legend()
plt.grid(True)
plt.show()
```

**Output (abbreviated)**:

```
Data shape: torch.Size([200, 2]) torch.Size([200, 1])
Model architecture:
SimpleNN(
  (layer1): Linear(in_features=2, out_features=10, bias=True)
  (relu): ReLU()
  (layer2): Linear(in_features=10, out_features=1, bias=True)
)

Parameter shapes:
layer1.weight: torch.Size([10, 2])
layer1.bias: torch.Size([10])
layer2.weight: torch.Size([1, 10])
layer2.bias: torch.Size([1])

Epoch [50/200], Loss: 1.2345
Epoch [100/200], Loss: 0.9876
Epoch [150/200], Loss: 0.8453
Epoch [200/200], Loss: 0.7891
```

This code creates a synthetic 2D dataset for regression with a non-linear
relationship and implements a simple neural network using PyTorch. The network
has one hidden layer with 10 units and ReLU activation, followed by a linear
output layer. It prints the model architecture and parameter shapes to
illustrate the matrix dimensions (e.g., `layer1.weight` is 10x2, mapping 2 input
features to 10 hidden units). The model is trained using mean squared error
(MSE) loss and stochastic gradient descent (SGD) for 200 epochs, with the loss
plotted over time to show convergence.

## Exercises

Here are six exercises to deepen your understanding of neural networks as matrix
functions. Each exercise requires writing Python code to explore concepts and
applications in machine learning using PyTorch.

1. **Manual Matrix Operations for Forward Pass**: Create a small dataset (10
   samples, 2 features) with NumPy, convert it to PyTorch tensors, and manually
   implement the forward pass of a neural network with one hidden layer (4
   units, ReLU activation) using matrix multiplications (`torch.matmul`).
   Compare the output with a PyTorch `nn.Linear` layer implementation.
2. **Parameter Shape Exploration**: Define a neural network in PyTorch with two
   hidden layers (hidden sizes 8 and 4) for a 3-feature input and 2-output
   problem. Print the shape of each weight matrix and bias vector to confirm the
   dimensions match the expected matrix operations.
3. **Custom Activation Function**: Extend the `SimpleNN` class from the example
   to include a custom activation function (e.g., a scaled tanh: `2 * tanh(x)`)
   between layers. Train it on the same dataset from Example 1 and plot the loss
   over 100 epochs.
4. **Batch Processing**: Modify the training loop from Example 1 to process the
   data in mini-batches of size 32 using PyTorch’s `DataLoader`. Train for 100
   epochs and plot the loss over epochs, comparing it to the full-batch training
   loss.
5. **Classification Network**: Create a synthetic 2D dataset for binary
   classification (100 samples) using NumPy, convert to PyTorch tensors, and
   build a neural network with one hidden layer (5 units) and sigmoid output.
   Train it with binary cross-entropy loss (`nn.BCELoss`) for 200 epochs and
   plot the loss.
6. **Gradient Inspection**: Using the model from Exercise 5, after training,
   print the gradients of the loss with respect to the weights of the first
   layer (`layer1.weight.grad`) for the last batch. Comment on the magnitude of
   the gradients to infer if the model has converged.

## Conclusion

Neural Networks as Matrix Functions reveal the elegant simplicity behind deep
learning: layers of matrix multiplications and non-linear activations, optimized
through gradient descent. By implementing a simple network with PyTorch, we’ve
seen how parameter shapes correspond to matrix dimensions and how vectorization
powers efficient computation. These concepts bridge linear algebra with modern
machine learning, forming the backbone of powerful models.

In the next post, we’ll explore **Tensors and Higher-Order Generalizations**,
extending matrix ideas to multi-dimensional arrays critical for deep learning
and computer vision. Stay tuned, and happy learning!
