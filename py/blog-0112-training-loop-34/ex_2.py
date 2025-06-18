import torch
import torch.nn as nn
import matplotlib.pyplot as plt

depth = 20

class DeepLayerNormSigmoidNet(nn.Module):
    def __init__(self, depth: int, hidden_size: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            ) for _ in range(depth)
        ])
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

net_ln = DeepLayerNormSigmoidNet(depth=depth)
x = torch.randn(16, 32)
target = torch.randn(16, 32)
output = net_ln(x)
loss = (output - target).pow(2).mean()
loss.backward()
grad_norms_ln = [layer[0].weight.grad.norm().item() for layer in net_ln.layers] # type: ignore

plt.plot(range(1, depth+1), grad_norms_ln, marker='*', color='purple')
plt.xlabel('Layer')
plt.ylabel('Weight Gradient Norm')
plt.title('Gradient Norms with LayerNorm')
plt.show()
