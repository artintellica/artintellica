import torch
import matplotlib.pyplot as plt
import numpy as np

# Set device to MPS if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Binomial distribution parameters
n = 10  # Number of trials
p = 0.4  # Probability of success
binomial_dist = torch.distributions.Binomial(total_count=n, probs=torch.tensor(p, device=device))

# Values to evaluate PMF (k = 0, 1, ..., n)
x = torch.arange(0, n + 1, dtype=torch.float32, device=device)

# Compute PMF using log_prob for numerical stability
log_pmf = binomial_dist.log_prob(x)
pmf = torch.exp(log_pmf).cpu().numpy()  # Exponentiate to get PMF, move to CPU for plotting

# Print PMF values
print(f"PMF for Binomial (n={n}, p={p}):")
for k, prob in enumerate(pmf):
    print(f"P(X={k}) = {prob:.4f}")

# Plot PMF as a stem plot
plt.stem(x.cpu().numpy(), pmf, basefmt=" ", linefmt='b-', markerfmt='bo')
plt.title(f"Binomial Distribution (n={n}, p={p})")
plt.xlabel("Number of Successes (k)")
plt.ylabel("PMF")
plt.xticks(np.arange(0, n + 1, 2))
plt.ylim(0, max(pmf) * 1.2)  # Adjust y-axis for visibility
plt.grid(True, alpha=0.3)
plt.show()
