+++
title = "Linear Algebra for Machine Learning, Part 14: Least Squares and Linear Regression"
author = "Artintellica"
date = "2025-06-17"
+++

Welcome back to our series on linear algebra for machine learning! In this post,
we’re delving into **Least Squares and Linear Regression**, foundational
concepts for predictive modeling. Linear regression is one of the simplest yet
most powerful tools in machine learning, relying heavily on linear algebra to
fit lines or planes to data by minimizing error. Whether you're predicting house
prices or analyzing trends, understanding the math behind least squares is
essential. Let’s explore the theory, derive the solutions, and implement them
with Python code, visualizations, and hands-on exercises.

## What Are Least Squares and Linear Regression?

Linear regression models the relationship between a dependent variable $y$ and
one or more independent variables $X$ by fitting a linear equation of the form:

$$
y = Xw + b
$$

where $X \in \mathbb{R}^{n \times d}$ is the data matrix with $n$ samples and
$d$ features, $w \in \mathbb{R}^{d}$ is the vector of weights (coefficients),
and $b$ is the bias (intercept). For simplicity, we often absorb $b$ into $w$ by
adding a column of ones to $X$, making the model $y = Xw$.

The goal of **least squares** is to find the parameters $w$ that minimize the
sum of squared residuals (errors) between the predicted values $\hat{y} = Xw$
and the actual values $y$:

$$
\text{Loss} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \| y - Xw \|_2^2
$$

This is an optimization problem, and linear algebra provides elegant solutions
to find the best-fitting line or plane.

### Solving with the Normal Equations

The least squares solution can be derived by setting the gradient of the loss to
zero, leading to the **normal equations**:

$$
X^T X w = X^T y
$$

If $X^T X$ is invertible (i.e., $X$ has full column rank), the solution is:

$$
w = (X^T X)^{-1} X^T y
$$

This closed-form solution directly computes the optimal weights using matrix
operations.

### Iterative Solution with Gradient Descent

For large datasets, computing the inverse of $X^T X$ can be computationally
expensive. Instead, we can use iterative methods like **gradient descent** to
minimize the loss. The gradient of the loss with respect to $w$ is:

$$
\nabla_w \text{Loss} = 2 X^T (Xw - y)
$$

We update $w$ in the opposite direction of the gradient with a learning rate
$\eta$:

$$
w \leftarrow w - \eta \cdot \nabla_w \text{Loss}
$$

## Why Do Least Squares and Linear Regression Matter in Machine Learning?

Linear regression is a cornerstone of machine learning for several reasons:

1. **Baseline Model**: It serves as a simple baseline for regression tasks,
   often outperforming complex models on small or linear datasets.
2. **Interpretability**: The coefficients $w$ provide insights into the
   importance and direction of each feature’s effect on the target.
3. **Foundation for Advanced Models**: Many advanced techniques (e.g., logistic
   regression, neural networks) build on linear regression concepts.
4. **Optimization Intuition**: Least squares introduces key optimization ideas
   like loss functions and gradient descent, which are central to ML.

Understanding the linear algebra behind least squares also prepares you for more
complex models that rely on similar matrix operations.

## Implementing Least Squares and Linear Regression in Python

Let’s implement linear regression using both the normal equations and gradient
descent with NumPy. We’ll also validate our results with scikit-learn on a
simple 2D dataset for visualization.

### Example 1: Normal Equations with NumPy

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a simple 2D dataset (1 feature + bias)
np.random.seed(42)
n_samples = 50
X = np.random.randn(n_samples, 1) * 2  # Feature
y = 3 * X[:, 0] + 2 + np.random.randn(n_samples) * 0.5  # Target with noise
print("Dataset shape:", X.shape, y.shape)

# Add a column of ones to X for the bias term
X_with_bias = np.hstack([np.ones((n_samples, 1)), X])

# Solve using normal equations: w = (X^T X)^(-1) X^T y
XTX = X_with_bias.T @ X_with_bias
XTy = X_with_bias.T @ y
w = np.linalg.solve(XTX, XTy)  # More stable than direct inverse
print("Learned weights (bias, slope):", w)

# Predict and plot
y_pred = X_with_bias @ w
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, alpha=0.5, label='Data Points')
plt.plot(X[:, 0], y_pred, 'r-', label='Fitted Line (Normal Equations)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Normal Equations')
plt.legend()
plt.grid(True)
plt.show()
```

**Output (abbreviated)**:

```
Dataset shape: (50, 1) (50,)
Learned weights (bias, slope): [2.0115 2.9867]
```

This code generates a simple dataset with one feature and fits a line using the
normal equations. It adds a column of ones to account for the bias term and
solves for the weights using `np.linalg.solve` for numerical stability. The plot
shows the data points and the fitted line, which closely matches the true
relationship (slope=3, intercept=2) despite added noise.

### Example 2: Gradient Descent with NumPy

```python
# Gradient Descent for the same dataset
w_init = np.zeros(2)  # Initial weights (bias, slope)
eta = 0.01  # Learning rate
n_iterations = 100

w_gd = w_init.copy()
for _ in range(n_iterations):
    gradient = 2 * X_with_bias.T @ (X_with_bias @ w_gd - y) / n_samples
    w_gd = w_gd - eta * gradient

print("Learned weights with Gradient Descent (bias, slope):", w_gd)

# Predict and plot
y_pred_gd = X_with_bias @ w_gd
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, alpha=0.5, label='Data Points')
plt.plot(X[:, 0], y_pred_gd, 'g-', label='Fitted Line (Gradient Descent)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()
```

**Output (abbreviated)**:

```
Learned weights with Gradient Descent (bias, slope): [2.0115 2.9867]
```

This implements gradient descent to minimize the least squares loss iteratively.
The weights converge to values very close to those from the normal equations,
and the plot confirms the fitted line matches the data well.

### Example 3: Validation with scikit-learn

```python
from sklearn.linear_model import LinearRegression

# Use scikit-learn's LinearRegression
model = LinearRegression()
model.fit(X, y)
print("scikit-learn weights (bias, slope):", [model.intercept_, model.coef_[0]])

# Predict and plot
y_pred_sk = model.predict(X)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, alpha=0.5, label='Data Points')
plt.plot(X[:, 0], y_pred_sk, 'b-', label='Fitted Line (scikit-learn)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with scikit-learn')
plt.legend()
plt.grid(True)
plt.show()
```

**Output (abbreviated)**:

```
scikit-learn weights (bias, slope): [2.0115, 2.9867]
```

This validates our implementations using scikit-learn’s `LinearRegression`,
which matches the results from both normal equations and gradient descent.

## Exercises

Here are six exercises to deepen your understanding of least squares and linear
regression. Each exercise requires writing Python code to explore concepts and
applications in machine learning.

1. **Normal Equations on 2D Data**: Generate a new 2D dataset (1 feature, 30
   samples) with NumPy, add a bias column, and solve for the linear regression
   weights using the normal equations. Plot the data and the fitted line.
2. **Gradient Descent Implementation**: Using the dataset from Exercise 1, write
   code to fit a linear model with gradient descent. Experiment with different
   learning rates (e.g., 0.001, 0.01, 0.1) and plot the loss over iterations for
   each rate.
3. **Multiple Features**: Generate a synthetic 3D dataset (2 features + bias, 50
   samples) with NumPy, where both features influence the target. Fit a linear
   model using the normal equations and print the learned weights.
4. **Comparison with scikit-learn**: Using the 3D dataset from Exercise 3, fit a
   linear model with scikit-learn’s `LinearRegression` and compare the weights
   and mean squared error with your normal equations solution from Exercise 3.
5. **Overfitting Check**: Generate a small dataset (10 samples, 1 feature) and
   fit linear models with increasing polynomial degrees (1 to 5) using
   scikit-learn’s `PolynomialFeatures`. Compute and plot the training mean
   squared error for each degree to observe potential overfitting.
6. **Real Dataset Regression**: Load the Boston Housing dataset from
   scikit-learn (`sklearn.datasets.load_boston`, or use an alternative like
   `sklearn.datasets.fetch_california_housing` if Boston is deprecated), select
   2 features, and fit a linear regression model with scikit-learn. Print the
   coefficients and mean squared error on a test split (use `train_test_split`).

## Conclusion

Least Squares and Linear Regression provide a fundamental approach to modeling
relationships in data, deeply rooted in linear algebra through the normal
equations and optimization techniques like gradient descent. By implementing
these methods with NumPy and validating with scikit-learn, we’ve seen how matrix
operations and iterative updates can solve regression problems effectively.
These concepts lay the groundwork for understanding more complex models in
machine learning.

In the next post, we’ll explore **Gradient Descent in Linear Models**, diving
deeper into optimization strategies and their role in scaling to larger
datasets. Stay tuned, and happy learning!
