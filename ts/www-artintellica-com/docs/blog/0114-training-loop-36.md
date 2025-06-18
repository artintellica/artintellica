+++
title = "Learn the Training Loop with PyTorch, Part 3.6: Monitoring and Debugging the Training Loop"
author = "Artintellica"
date = "2025-06-18"
model = "o3"
userDelimiter = "**USER:**"
assistantDelimiter = "**ASSISTANT:**"
+++

**USER:**

i need your help to write the next blog post in a series i am producing called
"Learn the Training Loop with PyTorch".

first, let me give you the outline of the series:

---

## **Module 1: The Elementary Training Loop**

**Goal:** Build foundational intuition by hand-coding a simple training loop
from scratch.

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

**Goal:** Extend the basic loop to more powerful models; introduce neural
network concepts.

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

**Goal:** Connect foundational understanding to the large-scale models used in
industry.

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

- Early stopping, batch/layer normalization, data augmentation (conceptual
  intro)

**8. The Training Loop in Practice: Case Studies**

- High-level overview of how OpenAI/Google/Meta might train LLMs: what’s the
  same, what’s different.
- What "scaling up" really means: dataset, parameters, hardware, tricks.

**9. Conclusion: What's Next After the Training Loop?**

- Discussion: fine-tuning, transfer learning, and unsupervised/self-supervised
  methods.

---

### **General Tips for Each Module:**

- **Start simple, build incrementally:** Keep code simple at first, then show
  how to layer complexity.
- **Accompany every lesson with practical code examples.**
- **Focus on intuition:** Use diagrams, plots, and hands-on demo to make
  concepts tangible.
- **Summarize and review regularly:** Add checklists or quizzes at the end of
  each module.

---

great. now, for each blog post, i am hoping to follow the following structure:

- the course is called "Learn the Training Loop with PyTorch"
- always provide a brief introduction to the topic at hand
- then always provide a very simple "ELI5" explanation of the topic, that
  theortically anyone could understand
- then provide the mathematical foundations. for math, always use dollar
  delimited LaTeX, and always provide a brief explanation of the math. for
  instance, $\mathbf{y} = \mathbf{X}\mathbf{w} + \mathbf{b}$ is the equation for
  linear regression, where $\mathbf{y}$ is the output vector, $\mathbf{X}$ is
  the input matrix, $\mathbf{w}$ is the weight vector, and $\mathbf{b}$ is the
  bias vector.
- for block-level math, always use two dollar signs, like this:
  $$
  \mathbf{y} = \mathbf{X}\mathbf{w} + \mathbf{b}
  $$
- after the mathematical foundations, such as definitions, then always provide
  an in-depth explanation, covering the topic in detail, and expanding on the
  explanation of the math if necessary.
- for other formatting, use markdown like headings, bullet points, and code
  blocks to make the content readable. do not use HTML tags.
- after the math is complete, then provide one or more demonstrations of the
  topic using PyTorch. if possible, always use PyTorch, not numpy, just to keep
  it simple and consistent. you can also use matplotlib to visualize the
  results.
- for all python code, always use type hints, and, if necessary, describe
  installation instructions using "uv" (not pip).
- if you need to use a gpu, note that the user may use apple, or nvidia, or no
- gpu at all, meaning in pytorch you should support cuda or mps or cpu.
- fallback from cuda to mps to cpu.
- then, create a series of exercises, somewhere from 1 - 4 total exercises that
  cover the material. please provide the soruce code for each exercise to make
  sure the student can easily compare their solution to the correct answer. do
  not hide the exercises and do not put HTML in the exercises.
- finally, provide a summary of the topic, and a list of key takeaways.

that's it, now can you please write blog post 3.6?

the description is:

**6. Monitoring and Debugging the Training Loop**
   - Visualizing loss/accuracy, diagnosing common problems

please respond with the full blog post only, and no other information.

**ASSISTANT:**

# Learn the Training Loop with PyTorch  
## Module 3.6: Monitoring and Debugging the Training Loop

---

Welcome to Module 3, Lesson 6 of "Learn the Training Loop with PyTorch"! At this point, you've learned how to build, improve, and scale up deep learning training loops. But even experienced practitioners know: debugging and monitoring are essential to successful model training.

In this blog post, we’ll demystify how to **visually track your training metrics, spot when things are going wrong, and quickly diagnose and resolve common problems** in the training loop using PyTorch.

---

## ELI5: Why Do We Monitor and Debug Training?

Imagine trying to bake a cake with no timer or oven window, and you have no idea if it’s burning or even cooking! Training a neural network is similar: **if you don’t check on what’s happening during training, you might waste hours on a model that’s not improving—or worse, is getting worse.**

Monitoring your training loop means **plotting things like "how wrong the model is" (loss) or "how often it gets the answer right" (accuracy) as it learns**. Debugging means figuring out why these numbers might not behave as you expect and fixing any mistakes.

---

## Mathematical Foundations

Let’s make sure we understand what we're tracking and why:

### Loss

The **loss function** measures how well the model's predictions match the target data.

For classification, a common loss is *cross-entropy*:
$$
\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \cdot \log(\hat{y}_{i,c})
$$

- $N$: number of examples
- $C$: number of classes
- $y_{i,c}$: 1 if example $i$ belongs to class $c$, 0 otherwise (one-hot)
- $\hat{y}_{i,c}$: model’s predicted probability for class $c$ (usually output of softmax)

### Accuracy

**Accuracy** measures the fraction of correct predictions:
$$
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}}
$$

### Why Plot Loss and Accuracy?

- **Loss** going down: The model is learning!
- **Accuracy** going up: The model is predicting correctly more often!
- **If not?** Your learning rate might be wrong, the data might not be set up correctly, or there could be a bug in the code.

---

## In-Depth: Diagnosing Common Training Loop Problems

Let’s dive into what can go wrong—and what plots can tell you:

1. **Loss does not decrease:**  
   - Possible causes: Learning rate too high/low, model architecture, data issues.
2. **Training accuracy high, validation accuracy low:**  
   - Overfitting.
3. **Accuracy not better than random guessing:**  
   - Data labels mixed up, bug in metric calculation, model underpowered.
4. **Loss is NaN (Not a Number):**  
   - Exploding gradients or bad data (division by zero, log(0)).

### How Plots Help

- **Loss curve flat:** Potential bug in gradient computation.
- **Validation loss lower than training loss:** Probably using test data for training; double-check your data split.
- **Accuracy volatile:** Might have too high a learning rate or insufficient data shuffling.

**Visualizing these metrics helps see the problem immediately—much faster than reading logs.**

---

## Demonstration: Visualizing Loss and Accuracy in PyTorch

Let's code an example showing how to track and plot these metrics for a simple classification task.

### Installation

To run this code, ensure you have PyTorch, torchvision, and matplotlib installed:

```bash
uv pip install torch torchvision matplotlib
```

### Example: Visualizing Training & Validation Metrics

```python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import List, Tuple

# Select device: cuda > mps > cpu
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()

# Simple MLP for MNIST classification
class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Load MNIST data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Initialize
model = SimpleMLP(28*28, 128, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Track metrics
train_losses: List[float] = []
val_losses: List[float] = []
train_accuracies: List[float] = []
val_accuracies: List[float] = []

def compute_accuracy(loader: DataLoader, model: nn.Module) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

num_epochs = 8
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Validation
    val_loss = 0.0
    val_steps = 0
    model.eval()
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            outputs = model(val_images)
            loss = criterion(outputs, val_labels)
            val_loss += loss.item() * val_images.size(0)
            val_steps += val_images.size(0)
    val_loss /= val_steps
    val_losses.append(val_loss)
    val_acc = compute_accuracy(val_loader, model)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}: "
          f"Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.4f}, "
          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

# Plotting
epochs = list(range(1, num_epochs + 1))
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, train_accuracies, label="Train Acc")
plt.plot(epochs, val_accuracies, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Epochs")
plt.legend()
plt.show()
```

**What You Should See:**  
- **Loss curves** should slope downward as training progresses.
- **Accuracy curves** should slope upward.
- If validation curves depart drastically from training, it's a sign of overfitting or another issue.

---

## Exercises

**Exercise 1 — Buggy Loss Calculation:**
Try deliberately switching `labels` and `outputs` in your loss calculation (i.e., `loss = criterion(labels, outputs)`). What happens to your loss and accuracy curves? Correct the bug and explain why.

```python
# WRONG
loss = criterion(labels, outputs)  # This is not the right argument order!

# CORRECT
loss = criterion(outputs, labels)  # outputs first, then labels
```

**Exercise 2 — No Shuffle, No Learning?**  
Set `shuffle=False` in the `DataLoader` for training data. What effect does this have on the loss and accuracy plots? Restore shuffling and compare.

```python
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
# TRY IT, then set shuffle=True again.
```

**Exercise 3 — Plotting Only Training Metrics**  
Remove all validation metrics tracking and plotting. What new issues arise if you can't see validation loss/accuracy? Why is tracking both crucial for effective debugging?

```python
# Remove val_losses and val_accuracies lists,
# don't plot val metrics in the final plot.
```

**Exercise 4 — Spotting Overfitting**  
Increase your model size (e.g., `hidden_dim=256`) but train only for 1-2 epochs with a small training dataset (e.g., 2000 samples). What do your validation and training curves look like? Is this overfitting or underfitting?

```python
# Use a subset for training:
subset_indices = list(range(2000))
small_train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
train_loader = DataLoader(small_train_dataset, batch_size=64, shuffle=True)

# Increase hidden_dim, reduce epochs.
model = SimpleMLP(28*28, 256, 10).to(device)
num_epochs = 2
```

---

## Summary

Monitoring and debugging the training loop is your *window into the model’s learning process*. By tracking and plotting loss and accuracy, you spot issues early, make informed adjustments, and ensure you’re actually teaching your model what you think you are.

---

## Key Takeaways

- **Always monitor loss and accuracy for both training and validation data.**
- **Downward loss, upward accuracy = model is learning.**
- **Divergence between train and validation: check for overfitting/bugs.**
- **Visualization makes it easy to see issues instantly.**
- **Misconfigured metrics or buggy code can destroy learning—always validate!**

Next up, we’ll explore **modern regularization and generalization techniques** to help your models learn reliably, beyond just monitoring their progress. See you in the next blog!

**USER:**


