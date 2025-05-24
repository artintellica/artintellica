import torch
import matplotlib.pyplot as plt
import numpy as np

# Set device to MPS if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Bernoulli distribution parameters
p = 0.3  # Probability of success
bernoulli_dist = torch.distributions.Bernoulli(probs=torch.tensor(p, device=device))

# Values to evaluate PMF (x = 0, 1)
x = torch.tensor([0.0, 1.0], device=device)  # Use float for log_prob compatibility

# Compute PMF using log_prob for numerical stability
log_pmf = bernoulli_dist.log_prob(x)
pmf = torch.exp(log_pmf).cpu().numpy()  # Exponentiate to get PMF, move to CPU for plotting

# Print PMF values
print(f"P(X=0) = {pmf[0]:.4f}")
print(f"P(X=1) = {pmf[1]:.4f}")

# Plot PMF as a bar chart
plt.bar(x.cpu().numpy(), pmf, color='blue', alpha=0.7, width=0.4)
plt.title(f"Bernoulli Distribution (p={p})")
plt.xlabel("x")
plt.ylabel("PMF")
plt.xticks([0, 1])
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.show()
