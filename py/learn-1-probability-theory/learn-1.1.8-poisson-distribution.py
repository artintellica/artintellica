import torch
import matplotlib.pyplot as plt
import numpy as np

# Set device to MPS if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Poisson distribution parameters
lam = 4.0  # Average rate (lambda)
poisson_dist = torch.distributions.Poisson(rate=torch.tensor(lam, device=device))

# Values to evaluate PMF (k = 0, 1, ..., 15)
x = torch.arange(0, 16, dtype=torch.float32, device=device)

# Compute PMF using log_prob for numerical stability
log_pmf = poisson_dist.log_prob(x)
pmf = torch.exp(log_pmf).cpu().numpy()  # Exponentiate to get PMF, move to CPU for plotting

# Print PMF values
print(f"PMF for Poisson (λ={lam}):")
for k, prob in enumerate(pmf):
    print(f"P(X={k}) = {prob:.4f}")

# Plot PMF as a stem plot
plt.stem(x.cpu().numpy(), pmf, basefmt=" ", linefmt='g-', markerfmt='go')
plt.title(f"Poisson Distribution (λ={lam})")
plt.xlabel("Number of Events (k)")
plt.ylabel("PMF")
plt.xticks(np.arange(0, 16, 2))
plt.ylim(0, max(pmf) * 1.2)  # Adjust y-axis for visibility
plt.grid(True, alpha=0.3)
plt.show()
