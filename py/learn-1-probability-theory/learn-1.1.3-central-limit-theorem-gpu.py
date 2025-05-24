import torch
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
n_samples = 1000
n_trials = 50
uniform_samples = torch.rand((n_samples, n_trials), device=device)  # Uniform [0,1]
sums = uniform_samples.sum(dim=1) / n_trials  # Average of sums
plt.hist(sums.cpu().numpy(), bins=30, density=True)
plt.title("Central Limit Theorem: Histogram of Averages")
plt.show()
