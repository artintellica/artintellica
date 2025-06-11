import torch
import matplotlib.pyplot as plt

# ### **Exercise 1:** Generate Binary Classification Data

# - Make two clouds in 2D using Gaussians with means $[0, 0]$ and $[3, 2]$ and a
#   shared covariance.
# - Stack them to form $N=100$ dataset and make integer labels $0$ and $1$.
# - Plot the dataset, color by class.
mean0 = torch.tensor([0.0, 0.0])
mean1 = torch.tensor([3.0, 2.0])
cov = torch.tensor([[1.0, 0.5], [0.5, 1.2]])
L = torch.linalg.cholesky(cov)
N = 100
X0 = torch.randn(N // 2, 2) @ L.T + mean0
X1 = torch.randn(N // 2, 2) @ L.T + mean1
X = torch.cat([X0, X1], dim=0)
y = torch.cat([torch.zeros(N // 2), torch.ones(N // 2)])
plt.scatter(X0[:, 0], X0[:, 1], color="b", alpha=0.5, label="Class 0")
plt.scatter(X1[:, 0], X1[:, 1], color="r", alpha=0.5, label="Class 1")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Binary Classification Data")
plt.grid(True)
plt.show()
