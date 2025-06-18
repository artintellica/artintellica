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
