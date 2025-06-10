import torch
import matplotlib.pyplot as plt

# Let's create a vector of sine values
t: torch.Tensor = torch.linspace(0, 2 * torch.pi, 100)
sin_t: torch.Tensor = torch.sin(t)

plt.figure(figsize=(8, 4))
plt.plot(t.numpy(), sin_t.numpy())
plt.title("Sine Wave")
plt.xlabel("t")
plt.ylabel("sin(t)")
plt.grid(True)
plt.show()
