import torch
import matplotlib.pyplot as plt

# Create a PyTorch vector containing $100$ linearly spaced points between $0$ and $4\pi$.
t: torch.Tensor = torch.linspace(0, 4 * torch.pi, 100)
t_cos: torch.Tensor = torch.cos(t)

plt.figure(figsize=(8, 4))
plt.plot(t.numpy(), t_cos.numpy())
plt.title("Cosine Wave")
plt.xlabel("t")
plt.ylabel("cos(t)")
plt.grid(True)
plt.show()
