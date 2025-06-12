import torch
import torchvision
from torch.utils.data import DataLoader

transform = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_set = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128)

import torch.nn as nn
import torch.nn.functional as F


class MiniMnistNet(nn.Module):
    def __init__(self, hidden: int = 100) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
plt.title("MNIST Confusion Matrix")
plt.show()

# Show misclassified digits
wrong_idx = np.where(all_preds != all_labels)[0]
plt.figure(figsize=(8, 2))
for i, idx in enumerate(wrong_idx[:8]):
    img = test_set[idx][0].squeeze(0).numpy()
    plt.subplot(1, 8, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"P{all_preds[idx]},T{all_labels[idx]}")
    plt.axis("off")
plt.suptitle("Misclassified")
plt.show()
