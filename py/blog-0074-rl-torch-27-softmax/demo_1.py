import torch
import matplotlib.pyplot as plt

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
for i in range(3):
    plt.scatter(
        X_list[i][:, 0], X_list[i][:, 1], color=colors[i], alpha=0.5, label=f"Class {i}"
    )
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Synthetic 3-Class Data")
plt.show()
