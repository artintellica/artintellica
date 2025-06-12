import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

z: torch.Tensor = torch.linspace(-5, 5, 200)
sigmoid: torch.Tensor = torch.sigmoid(z)
tanh: torch.Tensor = torch.tanh(z)
relu: torch.Tensor = F.relu(z)
leaky_relu: torch.Tensor = F.leaky_relu(z, negative_slope=0.1)
plt.plot(z.numpy(), sigmoid.numpy(), label='Sigmoid')
plt.plot(z.numpy(), tanh.numpy(), label='Tanh')
plt.plot(z.numpy(), relu.numpy(), label='ReLU')
plt.plot(z.numpy(), leaky_relu.numpy(), label='LeakyReLU')
plt.legend(); plt.xlabel('z'); plt.ylabel('Activation(z)')
plt.title("Activation Functions"); plt.grid(True); plt.show()

