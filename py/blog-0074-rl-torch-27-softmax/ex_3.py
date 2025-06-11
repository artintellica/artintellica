import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ### **Exercise 1:** Generate Synthetic Data for Three Classes

# - Create three 2D Gaussian blobs with different means.
# - Stack into features $X$ and labels $y$.
# - Plot, color-coded by class.
torch.manual_seed(42)
N = 300
cov = torch.tensor([[1.2, 0.8], [0.8, 1.2]])
L = torch.linalg.cholesky(cov)
means = [torch.tensor([-2.0, 0.0]), torch.tensor([2.0, 2.0]), torch.tensor([0.0, -2.0])]
X_list = []
y_list = []
for i, mu in enumerate(means):
    Xi = torch.randn(N // 3, 2) @ L.T + mu
    X_list.append(Xi)
    y_list.append(torch.full((N // 3,), i))
X = torch.cat(X_list)
y = torch.cat(y_list).long()
colors = ["b", "r", "g"]
# for i in range(3):
#     plt.scatter(
#         X_list[i][:, 0], X_list[i][:, 1], color=colors[i], alpha=0.5, label=f"Class {i}"
#     )
# plt.legend()
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.title("Synthetic 3-Class Data")
# plt.show()

# ### **Exercise 2:** Implement Softmax and Cross-Entropy Loss Manually

# - Implement the softmax function on a batch of logits.
# - Write manual cross-entropy loss for integer target labels.
# - Test on a toy example and compare with PyTorch’s `F.cross_entropy`.
def softmax(logits: torch.Tensor) -> torch.Tensor:
    # For numerical stability, subtract max
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits)
    return exp_logits / exp_logits.sum(dim=1, keepdim=True)
def cross_entropy_manual(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: (N, K), targets: (N,)
    N = logits.shape[0]
    log_probs = F.log_softmax(logits, dim=1)
    return -log_probs[torch.arange(N), targets].mean()
# Toy example
logits = torch.tensor([[2.0, 0.5, -1.0], [0.0, 3.0, 0.5]])
targets = torch.tensor([0, 1])
probs = softmax(logits)
manual_loss = cross_entropy_manual(logits, targets)
print("Probabilities:\n", probs)
print("Manual cross-entropy loss:", manual_loss.item())
# Compare with PyTorch's built-in function
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, targets)
print("PyTorch cross-entropy loss:", loss.item())

# ### **Exercise 3:** Train a Multiclass Classifier with PyTorch’s `nn.CrossEntropyLoss`

# - Initialize a linear model (`W` and `b`).
# - Train for 400 epochs using SGD on your three-class data.
# - Plot the loss curve.
# Model: simple linear (no bias for simplicity)
W = torch.zeros(2, 3, requires_grad=True)  # (features, classes)
b = torch.zeros(3, requires_grad=True)
lr = 0.05
loss_fn = torch.nn.CrossEntropyLoss()
losses = []
for epoch in range(400):
    logits = X @ W + b  # (N, 3)
    loss = loss_fn(logits, y)
    loss.backward()
    with torch.no_grad():
        W -= lr * W.grad if W.grad is not None else 0
        b -= lr * b.grad if b.grad is not None else 0
    W.grad.zero_() if W.grad is not None else None
    b.grad.zero_() if b.grad is not None else None
    losses.append(loss.item())
    if epoch % 100 == 0 or epoch == 399:
        print(f"Epoch {epoch}: Cross-entropy loss={loss.item():.3f}")
plt.plot(losses)
plt.title("Multiclass Classifier Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)
plt.show()
