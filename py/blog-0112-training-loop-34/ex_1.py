import torch
import torch.nn as nn
import matplotlib.pyplot as plt

depth = 20

class ExplodingNet(nn.Module):
    def __init__(self, depth: int, hidden_size: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(depth)
        ])
        self.activation = nn.ReLU()
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=2.0) # type: ignore
            nn.init.zeros_(layer.bias) # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

net_expl = ExplodingNet(depth=depth)
x = torch.randn(16, 32)
target = torch.randn(16, 32)
output = net_expl(x)
loss = (output - target).pow(2).mean()
loss.backward()
grad_norms_expl = [layer.weight.grad.norm().item() for layer in net_expl.layers] # type: ignore

plt.plot(range(1, depth+1), grad_norms_expl, marker='x', color='red')
plt.xlabel('Layer')
plt.ylabel('Weight Gradient Norm')
plt.title('Exploding Gradients with Large Weight Initialization')
plt.show()
