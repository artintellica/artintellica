+++
title = "Learn Reinforcement Learning with PyTorch, Part 3.8: Mini-Project—MNIST Digit Classifier (Shallow NN)"
author = "Artintellica"
date = "2024-06-12"
+++

## Introduction

Let’s bring together the skills you’ve developed so far and apply them to an iconic deep learning task: **classifying handwritten digits** with the MNIST dataset. In this mini-project, you will:

- Download & load the MNIST dataset using PyTorch’s torchvision.
- Build and train a shallow neural network to recognize digits 0–9.
- Evaluate model performance with a confusion matrix.
- Visualize both correct and incorrect predictions directly from the images.

By running through these classic steps, you’ll learn core deep learning workflow skills transferrable to RL, vision, and more.

---

## Mathematics: Classification with Shallow Neural Nets

**MNIST** images are $28\times28$ grayscale images—each a vector $x\in\mathbb{R}^{784}$.

A simple neural classifier:
- **Input Layer:** Flattened image ($784$ features)
- **Hidden Layer:** E.g., $H=128$ units, ReLU activation
- **Output Layer:** $10$ units, softmax activation (one per digit)

**Forward pass:**
\[
\begin{align*}
h &= \mathrm{ReLU}(Wx + b) \\
\hat{y} &= \mathrm{softmax}(Wh + b)
\end{align*}
\]

**Loss:**  
Use cross-entropy loss for classification:
\[
L = -\sum_{k=0}^9 y_k \log \hat{y}_k
\]
where $y_k$ is the one-hot label.

**Accuracy:**  
\[
\mathrm{acc} = \frac{\text{\# correct predictions}}{\text{total predictions}}
\]

---

## Explanation: How the Math Connects to Code

- **Data loading:** Use PyTorch’s torchvision.datasets to download and access images and labels as Tensors.
- **Model:** Compose layers with `nn.Sequential` or subclass `nn.Module`. For a shallow net, often one hidden layer is enough for MNIST to reach 97%+ accuracy.
- **Training:** Use cross-entropy loss and an optimizer (e.g., Adam or SGD).
- **Evaluation:** Track accuracy per epoch. Build a confusion matrix for more insight—this shows which digits are misclassified.
- **Visualization:** Use matplotlib to display sample images, including which predictions were correct and which weren’t.

---

## Python Demonstrations

### Demo 1: Download & Load the MNIST Dataset

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Download & normalize
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)
```

---

### Demo 2: Build and Train a Shallow NN on MNIST

```python
import torch.nn as nn
import torch.nn.functional as F

class ShallowNet(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(28*28, hidden)
        self.fc2: nn.Linear = nn.Linear(hidden, 10)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)      # Flatten images
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model: ShallowNet = ShallowNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
train_accs = []
for epoch in range(num_epochs):
    model.train()
    running_corrects = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)
    acc = running_corrects / total
    train_accs.append(acc)
    print(f"Epoch {epoch+1}: train accuracy = {acc:.4f}")
```

---

### Demo 3: Plot Confusion Matrix of Predictions

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Get all predictions and labels from the test set
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
disp.plot(cmap='Blues')
plt.title("MNIST Test Confusion Matrix")
plt.show()
```

---

### Demo 4: Visualize Correctly and Incorrectly Classified Digits

```python
# Find correctly and incorrectly classified indices
correct_idx = np.where(all_preds == all_labels)[0]
wrong_idx   = np.where(all_preds != all_labels)[0]

# Show a few correct
plt.figure(figsize=(8,2))
for i, idx in enumerate(correct_idx[:8]):
    img = test_set[idx][0].squeeze(0).numpy()
    plt.subplot(1,8,i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Pred {all_preds[idx]}")
plt.suptitle("Correctly Classified")
plt.show()

# Show a few incorrect
plt.figure(figsize=(8,2))
for i, idx in enumerate(wrong_idx[:8]):
    img = test_set[idx][0].squeeze(0).numpy()
    plt.subplot(1,8,i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"P{all_preds[idx]},T{all_labels[idx]}")
plt.suptitle("Misclassified")
plt.show()
```

---

## Exercises

### **Exercise 1:** Download and Load the MNIST Dataset Using PyTorch

- Use `torchvision.datasets.MNIST` to load training and test datasets as tensors.
- Create train/test `DataLoader` objects.

---

### **Exercise 2:** Build and Train a Shallow NN on MNIST

- Implement a neural net with one hidden layer (ReLU) and an output softmax (10 classes).
- Train for a few epochs (3–8).
- Track and print out training accuracy (and optionally loss).

---

### **Exercise 3:** Plot Confusion Matrix of Predictions

- After training, use `sklearn.metrics.confusion_matrix` to compute the test set confusion matrix (true vs. predicted labels).
- Visualize with `ConfusionMatrixDisplay` or matplotlib’s imshow.

---

### **Exercise 4:** Visualize a Few Correctly and Incorrectly Classified Digits

- Identify indices of correct and incorrect predictions.
- Use `plt.imshow()` to display image tensors for a sample of each group, showing both predicted and true labels.

---

### **Sample Starter Code for Exercises**

```python
import torch
import torchvision
from torch.utils.data import DataLoader

transform = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128)

import torch.nn as nn
import torch.nn.functional as F

class MiniMnistNet(nn.Module):
    def __init__(self, hidden: int = 100) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28*28, hidden)
        self.fc2 = nn.Linear(hidden, 10)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = MiniMnistNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train
for epoch in range(5):
    net.train()
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    print(f"Epoch {epoch+1}, train acc: {correct/total:.4f}")

# Confusion matrix and visualization
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
all_preds, all_labels = [], []
net.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("MNIST Confusion Matrix"); plt.show()

# Show misclassified digits
wrong_idx = np.where(all_preds != all_labels)[0]
plt.figure(figsize=(8,2))
for i, idx in enumerate(wrong_idx[:8]):
    img = test_set[idx][0].squeeze(0).numpy()
    plt.subplot(1,8,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"P{all_preds[idx]},T{all_labels[idx]}")
    plt.axis('off')
plt.suptitle("Misclassified")
plt.show()
```

---

## Conclusion

You’ve now completed a classic ML mini-project—training, evaluating, and interpreting a shallow neural net digit classifier with PyTorch. You’re ready to scale up to deeper models or leverage these skills for RL agents with real, raw input. If you’ve never built a deep learning app before, this is a firm step into the practical world.

**Next:** We’ll continue building RL intuition by connecting your neural network skills to reinforcement learning’s “loop”—states, actions, rewards, and learning policies!

Stay curious and try tweaking your net or data to beat your own accuracy! See you in Part 4.1!
