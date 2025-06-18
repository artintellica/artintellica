+++
title = "Learn the Training Loop with PyTorch, Part 2.6: Regularization and Overfitting"
author = "Artintellica"
date = "2025-06-18"
model = "o3"
userDelimiter = "**USER:**"
assistantDelimiter = "**ASSISTANT:**"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0107-training-loop-26"
+++

**USER:**

i need your help to write the next blog post in a series i am producing called "Learn the Training Loop with PyTorch".

first, let me give you the outline of the series:

---

## **Module 1: The Elementary Training Loop**

**Goal:** Build foundational intuition by hand-coding a simple training loop from scratch.

**1. Introduction: What is a Training Loop?**
   - Definition and importance  
   - Loss, optimization, and learning in ML

**2. The Simplest Case: Linear Regression**
   - What is linear regression?
   - The mean squared error loss
   - Deriving the update rules manually

**3. Batch vs. Stochastic Gradient Descent**
   - Differences and practical implications  
   - Hands-on: See both in Python/numpy

**4. Visualizing the Loss Landscape**
   - Plotting loss as a function of parameters
   - Demo: Watching convergence by hand

**5. Numerical vs. Analytical Gradients**
   - How are gradients computed?  
   - Manual calculation and confirmation with numerical gradient

**6. Recap and Key Takeaways**

---

## **Module 2: The Training Loop for Neural Networks**

**Goal:** Extend the basic loop to more powerful models; introduce neural network concepts.

**1. From Linear to Nonlinear: Why Neural Networks?**
   - Multi-layer perceptrons (MLPs)  
   - Activation functions (ReLU, sigmoid, tanh)

**2. Forward and Backward Passes**
   - The chain rule of differentiation  
   - Backpropagation intuition (no need for full rigor yet)

**3. Implementing a Simple Neural Net from Scratch**
   - Single hidden layer network in numpy/Python
   - Coding forward, loss, backward, and parameter updates

**4. The Role of Activations**
   - Intuitive explanation and code examples
   - Effect on learning and expressiveness

**5. Mini-batching and Data Pipelines**
   - How batching speeds up and stabilizes learning

**6. Regularization and Overfitting**
   - Techniques: L2, dropout (conceptual intro)

**7. Recap: Comparing Our Simple Network with Linear Regression**

---

## **Module 3: Advanced Training Loops and Modern Tricks**

**Goal:** Connect foundational understanding to the large-scale models used in industry.

**1. Optimization Algorithms Beyond SGD**
   - Adam, RMSProp, etc.: How they work, why they're useful

**2. Learning Rate Scheduling**
   - How/why learning rates are changed during training  
   - Demo: Effect of learning rate on convergence and results

**3. Weight Initialization**
   - Why it matters, common strategies

**4. Deeper Networks and Backprop Challenges**
   - Vanishing/exploding gradients  
   - Solutions: normalization, skip connections

**5. Large-Scale Training: Data Parallelism and Hardware**
   - How the same training loop runs efficiently on GPUs and across many machines

**6. Monitoring and Debugging the Training Loop**
   - Visualizing loss/accuracy, diagnosing common problems

**7. Modern Regularization and Generalization Techniques**
   - Early stopping, batch/layer normalization, data augmentation (conceptual intro)

**8. The Training Loop in Practice: Case Studies**
   - High-level overview of how OpenAI/Google/Meta might train LLMs: what’s the same, what’s different.
   - What "scaling up" really means: dataset, parameters, hardware, tricks.

**9. Conclusion: What's Next After the Training Loop?**
   - Discussion: fine-tuning, transfer learning, and unsupervised/self-supervised methods.

---

### **General Tips for Each Module:**

- **Start simple, build incrementally:** Keep code simple at first, then show how to layer complexity.
- **Accompany every lesson with practical code examples.**
- **Focus on intuition:** Use diagrams, plots, and hands-on demo to make concepts tangible.
- **Summarize and review regularly:** Add checklists or quizzes at the end of each module.

---

great. now, for each blog post, i am hoping to follow the following structure:

- the course is called "Learn the Training Loop with PyTorch"
- always provide a brief introduction to the topic at hand
- then always provide a very simple "ELI5" explanation of the topic, that theortically anyone could understand
- then provide the mathematical foundations. for math, always use dollar delimited LaTeX, and always provide a brief explanation of the math. for instance, $\mathbf{y} = \mathbf{X}\mathbf{w} + \mathbf{b}$ is the equation for linear regression, where $\mathbf{y}$ is the output vector, $\mathbf{X}$ is the input matrix, $\mathbf{w}$ is the weight vector, and $\mathbf{b}$ is the bias vector.
- for block-level math, always use two dollar signs, like this:
  $$
  \mathbf{y} = \mathbf{X}\mathbf{w} + \mathbf{b}
  $$
- after the mathematical foundations, such as definitions, then always provide an in-depth explanation, covering the topic in detail, and expanding on the explanation of the math if necessary.
- for other formatting, use markdown like headings, bullet points, and code blocks to make the content readable. do not use HTML tags.
- after the math is complete, then provide one or more demonstrations of the topic using PyTorch. if possible, always use PyTorch, not numpy, just to keep it simple and consistent. you can also use matplotlib to visualize the results.
- for all python code, always use type hints, and, if necessary, describe installation instructions using "uv" (not pip).
- then, create a series of exercises, somewhere from 1 - 4 total exercises that cover the material. please provide the soruce code for each exercise to make sure the student can easily compare their solution to the correct answer. do not hide the exercises and do not put HTML in the exercises.
- finally, provide a summary of the topic, and a list of key takeaways.

that's it, now can you please write blog post 2.6?

the description is:

**6. Regularization and Overfitting**
   - Techniques: L2, dropout (conceptual intro)

please respond with the full blog post only, and no other information.

**ASSISTANT:**

# Learn the Training Loop with PyTorch  
## Module 2.6: Regularization and Overfitting

---

In this lesson, we dive into two crucial concepts for anyone training neural networks: **overfitting** and **regularization**. You'll learn what overfitting is, why it happens, and how regularization techniques like **L2 regularization** and **Dropout** help make your models more robust. We'll look at the intuition, the math, and practical PyTorch implementations to cement your understanding.

---

## ELI5: Why Do Neural Nets Make Mistakes After Seeming to Do Well?

Imagine you're preparing for a spelling bee. Each night, you memorize a list of words, and your friend quizzes you on exactly those. You ace the list every time! But in the contest, the words are different, and suddenly you're not so great.  
Your "model" (your brain) learned the training words perfectly, but didn’t really understand how to spell in general.

**That's overfitting.**  
Your neural network got so good at remembering the training data, it failed to generalize to new, unseen examples.

**Regularization is like practicing with different word lists, or being penalized for only memorizing.** It helps you *learn to spell*—not just recall answers.

---

## Mathematical Foundations

### What is Overfitting?

Overfitting occurs when a model learns not only the underlying patterns in the training data but also the noise, leading to poor performance on new data.

Suppose we have a model $f(\mathbf{x}; \boldsymbol\theta)$, which outputs predictions based on input $\mathbf{x}$ and parameters $\boldsymbol\theta$ (weights and biases).

We want to minimize the **loss** on the whole data distribution ($\mathcal{D}_{\text{all}}$), but we only have access to a **finite training set** ($\mathcal{D}_{\text{train}}$).

If the model capacity is too high, we may have:

$$
\text{Train Loss} \ll \text{Test Loss}
$$

- Where "**Train Loss**" = average loss on your training data  
- "**Test Loss**" = average loss on new, unseen data

### L2 Regularization (a.k.a. Ridge, Weight Decay)

L2 Regularization discourages the model weights from becoming too large, making the model simpler and less prone to memorization.

The standard loss (e.g., Mean Squared Error for regression):

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

With **L2 regularization**, the new objective becomes:

$$
\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{MSE}} + \lambda \sum_{j} w_j^2
$$

- $w_j$ = each weight parameter in the model
- $\lambda$ = regularization strength (a hyperparameter that you choose)

You can think of this as penalizing the model for having large weights.

### Dropout (Conceptual Math Only)

Dropout is a technique where, during training, some neurons are *randomly "dropped out"* (set to zero) for each mini-batch. This prevents neurons from co-adapting and forces the network to distribute its learning.

Mathematically, for activations $\mathbf{h}$ in a layer, with dropout probability $p$:

$$
\tilde{\mathbf{h}} = \mathbf{h} \odot \mathbf{m}
$$

- $\mathbf{m}$ is a random *mask* vector, where each element is $1$ with probability $(1 - p)$ and $0$ with probability $p$
- $\odot$ denotes element-wise multiplication

During **inference** (model evaluation), dropout is turned off, and activations are scaled accordingly.

---

## In-Depth Explanation: How Regularization Combats Overfitting

### Why Do Models Overfit?

Neural networks, especially deep ones, have many parameters (weights and biases). When given limited data, they can "cheat" by memorizing exact training examples, instead of learning true underlying patterns.

As a result, predictions on new (unseen) data are often unreliable.

### How Does L2 Regularization Work?

By adding a penalty on large weights (via the $\lambda$ term), L2 regularization encourages the network to keep weights small. Smaller weights often result in simpler network behavior, which tends to generalize better. If we make $\lambda$ too large, the network can't learn much and may **underfit** (do poorly on all data).

L2 works by modifying the gradients during backpropagation, slightly nudging weights toward zero as training progresses.

### How Does Dropout Work?

Dropout randomly "turns off" neurons during each training step. This means the network can't rely on any one neuron, and must spread the learning throughout the network.

- **Analogy:** Like forming a basketball team but sometimes requiring different players to sit out, the team must learn to play well together no matter who's on the court.

At inference time, all neurons are active, but their outputs are scaled to account for the fact that some were missing during training.

---

## Demonstration: Regularization in PyTorch

Let's train a simple neural network with and without regularization techniques to see the effect.

### Installation If Needed

```bash
uv pip install torch matplotlib
```

### Creating a Overfitting Toy Dataset

We'll make a simple one-dimensional regression problem with noisy labels, and intentionally keep the dataset small (to make overfitting easy).

```python
import torch
import matplotlib.pyplot as plt
from torch import nn
from typing import Tuple

def make_toy_data(n_samples: int=30, seed: int=42) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    X = torch.linspace(-3, 3, n_samples).reshape(-1, 1)
    # True function: y = sin(x), noise added
    y = torch.sin(X) + 0.3 * torch.randn_like(X)
    return X, y

X_train, y_train = make_toy_data()
X_test = torch.linspace(-3, 3, 100).reshape(-1, 1)
y_test = torch.sin(X_test)

plt.scatter(X_train.numpy(), y_train.numpy(), label='Train data')
plt.plot(X_test.numpy(), y_test.numpy(), label='True function', color='green')
plt.legend()
plt.show()
```

---

### 1. Neural Network Without Regularization

```python
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, 
          n_epochs: int, lr: float, l2: float=0.0, dropout: bool=False) -> list:
    if dropout:
        # Replace activation layers with Dropout+activation
        new_layers = []
        for layer in model.layers:
            new_layers.append(layer)
            if isinstance(layer, nn.Tanh):  # Insert Dropout after activation
                new_layers.append(nn.Dropout(p=0.3))
        model.layers = nn.Sequential(*new_layers)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.MSELoss()
    losses = []
    model.train()
    for epoch in range(n_epochs):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# Training without regularization
model_no_reg = TinyNet()
losses_no_reg = train(model_no_reg, X_train, y_train, n_epochs=500, lr=0.01)

plt.plot(losses_no_reg)
plt.title('Training Loss (No Regularization)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()

# Plot predictions
model_no_reg.eval()
with torch.no_grad():
    y_pred_train = model_no_reg(X_train)
    y_pred_test = model_no_reg(X_test)
plt.scatter(X_train.numpy(), y_train.numpy(), label='Train')
plt.plot(X_test.numpy(), y_test.numpy(), label='True', color='green')
plt.plot(X_test.numpy(), y_pred_test.numpy(), label='Predicted', color='red')
plt.legend()
plt.title('Model Fit (No Regularization)')
plt.show()
```

---

### 2. Neural Network **With L2 Regularization** (Weight Decay)

```python
model_l2 = TinyNet()
losses_l2 = train(model_l2, X_train, y_train, n_epochs=500, lr=0.01, l2=0.1)

plt.plot(losses_l2, label='Train Loss (L2)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss (L2 Regularization)')
plt.show()

# Plot predictions
model_l2.eval()
with torch.no_grad():
    y_pred_test_l2 = model_l2(X_test)
plt.scatter(X_train.numpy(), y_train.numpy(), label='Train')
plt.plot(X_test.numpy(), y_test.numpy(), label='True', color='green')
plt.plot(X_test.numpy(), y_pred_test_l2.numpy(), label='Predicted (L2)', color='red')
plt.legend()
plt.title('Model Fit (L2 Regularization)')
plt.show()
```

---

### 3. Neural Network **With Dropout**

```python
model_dropout = TinyNet()
losses_drop = train(model_dropout, X_train, y_train, n_epochs=500, lr=0.01, dropout=True)

plt.plot(losses_drop, label='Train Loss (Dropout)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss (Dropout)')
plt.show()

model_dropout.eval()
with torch.no_grad():
    y_pred_test_drop = model_dropout(X_test)
plt.scatter(X_train.numpy(), y_train.numpy(), label='Train')
plt.plot(X_test.numpy(), y_test.numpy(), label='True', color='green')
plt.plot(X_test.numpy(), y_pred_test_drop.numpy(), label='Predicted (Dropout)', color='red')
plt.legend()
plt.title('Model Fit (Dropout)')
plt.show()
```

---

### 4. Comparing All Three

Observe how unregularized, L2, and dropout models fit the data differently.

```python
plt.figure(figsize=(8,6))
plt.plot(X_test.numpy(), y_test.numpy(), label='True', color='green')
with torch.no_grad():
    plt.plot(X_test.numpy(), model_no_reg(X_test).numpy(), label='No Reg', linestyle='--')
    plt.plot(X_test.numpy(), model_l2(X_test).numpy(), label='L2', linestyle='-.')
    plt.plot(X_test.numpy(), model_dropout(X_test).numpy(), label='Dropout', linestyle=':')
plt.scatter(X_train.numpy(), y_train.numpy(), label='Train', alpha=0.5)
plt.legend()
plt.title('Regularization Comparison')
plt.show()
```

Notice that:
- The **unregularized** model may wiggle and fit every data point (overfitting).
- The **L2** and **Dropout** models produce smoother, more generalizable fits.

---

## Exercises

1. **Exercise 1: Vary L2 Regularization Strength**
   
   Modify the `l2` argument when training your model. Try `l2=0.001`, `0.01`, `0.1`, and `1.0`. Plot all the predicted functions.  
   What happens as you increase L2 regularization?

   **Solution:**
   ```python
   for l2 in [0.001, 0.01, 0.1, 1.0]:
       model = TinyNet()
       train(model, X_train, y_train, n_epochs=500, lr=0.01, l2=l2)
       with torch.no_grad():
           y_pred = model(X_test)
       plt.plot(X_test.numpy(), y_pred.numpy(), label=f'l2={l2}')
   plt.plot(X_test.numpy(), y_test.numpy(), label='True', color='green', linewidth=2)
   plt.scatter(X_train.numpy(), y_train.numpy(), color='black', alpha=0.3)
   plt.legend()
   plt.title('Effect of L2 Regularization')
   plt.show()
   ```

2. **Exercise 2: Try a Deeper Network With Dropout**

   Change `TinyNet` to have **4 hidden layers** instead of 2. Retrain the model with dropout enabled.  
   What do you observe versus the shallow network?

   **Solution:**
   ```python
   class DeepNet(nn.Module):
       def __init__(self):
           super().__init__()
           self.layers = nn.Sequential(
               nn.Linear(1, 64),
               nn.Tanh(),
               nn.Linear(64, 64),
               nn.Tanh(),
               nn.Linear(64, 64),
               nn.Tanh(),
               nn.Linear(64, 64),
               nn.Tanh(),
               nn.Linear(64, 1)
           )
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           return self.layers(x)

   deep_model = DeepNet()
   train(deep_model, X_train, y_train, n_epochs=500, lr=0.01, dropout=True)
   with torch.no_grad():
       y_pred_deep = deep_model(X_test)
   plt.plot(X_test.numpy(), y_test.numpy(), label='True', color='green')
   plt.plot(X_test.numpy(), y_pred_deep.numpy(), label='Dropout DeepNet', color='purple')
   plt.scatter(X_train.numpy(), y_train.numpy(), alpha=0.4)
   plt.legend()
   plt.title('Deeper Net with Dropout')
   plt.show()
   ```

3. **Exercise 3: Add Both L2 and Dropout**

   Modify the `train` function to accept both `l2` and `dropout` as arguments and run them together.  
   What do you see when both are used?

   **Solution:**
   (Function defined above already supports both.)
   ```python
   both_model = TinyNet()
   train(both_model, X_train, y_train, n_epochs=500, lr=0.01, l2=0.01, dropout=True)
   with torch.no_grad():
       y_pred_both = both_model(X_test)
   plt.plot(X_test.numpy(), y_test.numpy(), label='True', color='green')
   plt.plot(X_test.numpy(), y_pred_both.numpy(), label='L2 + Dropout', color='orange')
   plt.scatter(X_train.numpy(), y_train.numpy(), alpha=0.4)
   plt.legend()
   plt.title('L2 + Dropout Combined')
   plt.show()
   ```

---

## Summary

- **Overfitting** is when a model learns the training data too well—including its noise—leading to poor real-world performance.
- **Regularization** is essential for making models robust and generalizable.
- **L2 Regularization (Weight Decay):** Encourages smaller weights, resulting in simpler models less prone to overfitting.
- **Dropout:** Randomly disables neurons during training, forcing the network to learn redundant, robust features.
- Both can be used together for even better regularization.

---

### Key Takeaways

- **Overfitting** is a common challenge in deep learning, but can be detected by a large gap between training and test loss.
- **L2 Regularization** adds a penalty for large weights, smoothing the model’s predictions.
- **Dropout** promotes robustness by making the network "practice" with missing neurons.
- **Regularization is not a cure-all:** It’s a balancing act—too much leads to underfitting.
- PyTorch lets you apply both L2 and dropout with just a few lines of code.
- Experimentation is key: try different regularization strengths, dropout rates, and network sizes to develop intuition!

---

**Next up:** We'll recap Module 2 and see how our neural networks stack up against linear regression!

**USER:**

in this code i'm getting an error: model.layers is not iterable. how do i fix this?

```python
        for layer in model.layers:
            new_layers.append(layer)
            if isinstance(layer, nn.Tanh):  # Insert Dropout after activation
                new_layers.append(nn.Dropout(p=0.3))
```

**ASSISTANT:**

Great catch! The problem arises because in your code, `model.layers` is an instance of `nn.Sequential`. 
While you can't iterate directly over an `nn.Sequential` object like a regular Python list, you **can** iterate over its child modules using `model.layers.children()`. 

So, you should replace this:

```python
for layer in model.layers:
    new_layers.append(layer)
    if isinstance(layer, nn.Tanh):
        new_layers.append(nn.Dropout(p=0.3))
```

with this:

```python
for layer in model.layers.children():
    new_layers.append(layer)
    if isinstance(layer, nn.Tanh):  # Insert Dropout after activation
        new_layers.append(nn.Dropout(p=0.3))
```

Alternatively, you can use `model.layers` as a list by accessing its `_modules` attribute values (though this is a bit more "private"):

```python
for layer in model.layers._modules.values():
    # ... as above
```

**But the recommended way is:**
```python
for layer in model.layers.children():
    new_layers.append(layer)
    if isinstance(layer, nn.Tanh):
        new_layers.append(nn.Dropout(p=0.3))
```

This will resolve the error!

**USER:**

can you please give me the full source code for the final demonstration including all three approaches?

**ASSISTANT:**

Absolutely! Below is a **complete, runnable script** that:

- Generates a noisy (overfittable) regression dataset
- Defines a simple neural network (`TinyNet`)
- Implements training with/without L2 regularization and with Dropout
- Plots the results for all three approaches side by side

Copy-paste this code into a `.py` file and run it—you can tweak parameters as you wish!

---

```python
import torch
from torch import nn
import matplotlib.pyplot as plt
from typing import Tuple, List

# ==== 1. Data Generation ====
def make_toy_data(n_samples: int = 30, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    X = torch.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = torch.sin(X) + 0.3 * torch.randn_like(X)
    return X, y

X_train, y_train = make_toy_data()    # 30 points with noise
X_test = torch.linspace(-3, 3, 100).reshape(-1, 1)
y_test = torch.sin(X_test)            # true signal without noise

# ==== 2. Model Definition ====
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# ==== 3. Training Function ====
def train(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int,
    lr: float,
    l2: float = 0.0,
    dropout: bool = False,
    dropout_p: float = 0.3
) -> List[float]:
    if dropout:
        # insert Dropout after each activation
        new_layers = []
        for layer in model.layers.children():  # FIX: use .children()
            new_layers.append(layer)
            if isinstance(layer, nn.Tanh):
                new_layers.append(nn.Dropout(p=dropout_p))
        model.layers = nn.Sequential(*new_layers)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.MSELoss()
    losses = []
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# ==== 4. Training Three Models ====

# (A) No Regularization
model_no_reg = TinyNet()
losses_no_reg = train(model_no_reg, X_train, y_train, n_epochs=500, lr=0.01)

# (B) L2 Regularization (weight decay)
model_l2 = TinyNet()
losses_l2 = train(model_l2, X_train, y_train, n_epochs=500, lr=0.01, l2=0.1)

# (C) Dropout
model_dropout = TinyNet()
losses_dropout = train(model_dropout, X_train, y_train, n_epochs=500, lr=0.01, dropout=True, dropout_p=0.3)

# ==== 5. Plotting Loss Curves ====
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(losses_no_reg)
plt.title("No Regularization")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")

plt.subplot(1,3,2)
plt.plot(losses_l2)
plt.title("L2 Regularization")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")

plt.subplot(1,3,3)
plt.plot(losses_dropout)
plt.title("Dropout")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.tight_layout()
plt.show()

# ==== 6. Plotting Predictions ====
plt.figure(figsize=(8,6))
plt.plot(X_test.numpy(), y_test.numpy(), label='True Function', color='green', linewidth=2)
with torch.no_grad():
    plt.plot(X_test.numpy(), model_no_reg(X_test).numpy(), label='No Reg', linestyle='--', color='orange')
    plt.plot(X_test.numpy(), model_l2(X_test).numpy(), label='L2', linestyle='-.', color='blue')
    plt.plot(X_test.numpy(), model_dropout(X_test).numpy(), label='Dropout', linestyle=':', color='red')
plt.scatter(X_train.numpy(), y_train.numpy(), label='Train Data', color='black', alpha=0.5)
plt.legend()
plt.title("Regularization Comparison: Model Predictions")
plt.xlabel("x"); plt.ylabel("y")
plt.show()
```

---

**Notes:**  
- Requires `torch` and `matplotlib`. Install with `uv pip install torch matplotlib`
- You can freely adjust learning rates, number of epochs, L2 penalty, or Dropout probability.
- This code will display both loss curves and a single summary plot of all three models' fits.

Let me know if you want exercises or function refactors!

**USER:**


