+++
title = "Learn the Training Loop with PyTorch, Part 1.1: Introduction to the Training Loop and Linear Regression"
author = "Artintellica"
date = "2025-06-15"
+++

Welcome to **Learn the Training Loop with PyTorch**! In this first post, we'll take our very first steps into the heart of modern machine learning: **the training loop**. To make things as clear as possible, we'll use a classic and simple example—**linear regression**—and build everything from the ground up.

---

## ELI5: What is a Training Loop?

Imagine you're learning to shoot basketball hoops. The first time you try, you miss. You watch where your shot lands, figure out what went wrong, and adjust your aim for the next try. Each time, you shoot, check how close you were, and adjust.

That's exactly what a **training loop** is:  
- The model (the "player") makes a guess.  
- We measure how good or bad the guess was (like seeing where your shot landed: the "loss").  
- The model uses this information to adjust itself, aiming to get better with every try.

**Linear regression** is our simplest basketball hoop—it tries to draw a straight line through a set of points, learning to make better predictions by minimizing its mistakes each time.

---

## Mathematical Foundations

### The Linear Regression Model

Linear regression tries to fit a line (or hyperplane) to data. The basic equation is:

$$
\mathbf{y} = \mathbf{X}\mathbf{w} + \mathbf{b}
$$

- $\mathbf{y}$ is the vector of outputs (predicted values).
- $\mathbf{X}$ is the matrix of inputs (features/data points).
- $\mathbf{w}$ is the vector of weights the model learns.
- $\mathbf{b}$ is the bias (offset term).

### Loss Function

We need a way to measure how far off our predictions are. For linear regression, the usual choice is **Mean Squared Error (MSE)**:

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

- $N$ is the number of samples.
- $y_i$ is the true value.
- $\hat{y}_i$ is the predicted value.

---

## In-Depth Explanation

The goal of linear regression is to find weights $\mathbf{w}$ and bias $\mathbf{b}$ so that our predicted values ($\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + \mathbf{b}$) are as close as possible to the actual target values $\mathbf{y}$.

The **training loop**, step by step:
1. **Forward Pass**: Compute predictions with the current weights/bias.
2. **Compute Loss**: Calculate the loss (how bad are our predictions?).
3. **Backward Pass**: Compute gradients of the weights/bias with respect to the loss (how should we change them to lessen the error?).
4. **Update Parameters**: Adjust the weights/bias a bit to reduce the loss.
5. **Repeat**: Do this for many rounds (epochs or steps), getting a bit better each time.

By repeating this loop, the model gradually learns the best line.

---

## PyTorch Demonstration

Let's implement a minimal but complete linear regression training loop in PyTorch!

### Installation

If you haven't already installed the necessary libraries, run:
```bash
uv pip install torch matplotlib
```

### Code

```python
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from typing import Tuple, List

def generate_data(n_samples: int = 100) -> Tuple[Tensor, Tensor]:
    """Generate synthetic linear data with noise."""
    X = torch.linspace(-3, 3, n_samples).unsqueeze(1)  # shape: (n_samples, 1)
    true_w, true_b = 2.0, -1.0
    y = true_w * X + true_b + 0.5 * torch.randn_like(X)
    return X, y

def train_linear_regression(
    X: Tensor,
    y: Tensor,
    n_epochs: int = 200,
    lr: float = 0.05
) -> Tuple[Tensor, Tensor, List[float]]:
    # Initialize parameters
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    optimizer = torch.optim.SGD([w, b], lr=lr)
    losses = []

    for epoch in range(n_epochs):
        # Forward pass
        y_pred = X * w + b  # shape: (n_samples, 1)
        loss = torch.mean((y_pred - y) ** 2)
        losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # (Optional) Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

    return w.detach(), b.detach(), losses

def plot_results(X: Tensor, y: Tensor, w: Tensor, b: Tensor, losses: List[float]) -> None:
    plt.figure(figsize=(12, 5))

    # Plot data and fitted line
    plt.subplot(1, 2, 1)
    plt.scatter(X.numpy(), y.numpy(), label="Data")
    plt.plot(X.numpy(), (X * w + b).numpy(), color="red", label="Fitted Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Linear Regression Fit")

    # Plot loss over time
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Generate data
    X, y = generate_data()
    # Train the model
    w, b, losses = train_linear_regression(X, y)
    # Visualize the results
    plot_results(X, y, w, b, losses)
```

---

## Exercises

### 1. Change the Learning Rate

**Task:** Set the learning rate to a larger value (e.g., `lr=0.5`). What do you observe with the loss curve?

**Solution:**
```python
# Just call train_linear_regression with lr=0.5
w, b, losses = train_linear_regression(X, y, lr=0.5)
plot_results(X, y, w, b, losses)
# Observe: The loss might oscillate or diverge if lr is too high!
```

---

### 2. Add More Noise to the Data

**Task:** Modify the `generate_data` function to use `torch.randn_like(X) * 2.0` instead of `0.5`. How does this affect the fit and loss?

**Solution:**
```python
def generate_data(n_samples: int = 100) -> Tuple[Tensor, Tensor]:
    X = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    true_w, true_b = 2.0, -1.0
    y = true_w * X + true_b + 2.0 * torch.randn_like(X)
    return X, y

# Regenerate data and re-train
X, y = generate_data()
w, b, losses = train_linear_regression(X, y)
plot_results(X, y, w, b, losses)
# Observe: The fit is less "clean", and the loss is generally higher.
```

---

### 3. Try Using More Epochs

**Task:** Increase `n_epochs` from 200 to 1000. What happens to the loss and learned parameters?

**Solution:**
```python
w, b, losses = train_linear_regression(X, y, n_epochs=1000)
plot_results(X, y, w, b, losses)
# Observe: Loss decreases further for low-noise data, parameters become more accurate.
```

---

### 4. Predict New Data Points

**Task:** After training, use the learned model to predict $y$ for $X=4.0$.

**Solution:**
```python
X_new = torch.tensor([[4.0]])
y_pred_new = X_new * w + b
print(f"Prediction for X=4.0: {y_pred_new.item():.2f}")
```

---

## Summary and Key Takeaways

- **The training loop** is the core process of machine learning: make predictions, measure error, update, and repeat.
- **Linear regression** is a foundational model that illustrates the training loop clearly.
- **Loss functions** (like mean squared error) let us measure how good our model's guesses are.
- **Gradient descent** iteratively adjusts model parameters to minimize loss.
- With **PyTorch**, you can easily implement training loops—these same patterns power huge neural networks!
- Playing with parameters (learning rate, epochs, noise) helps you understand the dynamics of learning.

---

**Next up:** We'll dig deeper, exploring how the training loop extends to more interesting and complex machine learning models!
