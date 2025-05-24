import torch
import matplotlib.pyplot as plt
import numpy as np

# Set device to MPS (GPU) if available, else CPU
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

# Define Binomial distribution parameters
n = 10  # Number of trials
p = 0.4  # Probability of success per trial
binomial_dist = torch.distributions.Binomial(total_count=n, probs=torch.tensor(p, device=device))

# Calculate theoretical expected value and variance
expected_value = n * p  # E[X] = n * p
variance = n * p * (1 - p)  # Var(X) = n * p * (1 - p)
std_dev = np.sqrt(variance)  # Standard deviation = sqrt(variance)

# Generate samples to compute empirical expected value and variance
n_samples = 100000  # Number of samples
samples = binomial_dist.sample((n_samples,)).to(device)
empirical_mean = samples.mean().item()  # Empirical mean
empirical_variance = samples.var().item()  # Empirical variance

# Compute PMF for plotting
x = torch.arange(0, n + 1, dtype=torch.float32, device=device)  # Values k = 0 to n
log_pmf = binomial_dist.log_prob(x)  # Log-probabilities for stability
pmf = torch.exp(log_pmf).cpu().numpy()  # Exponentiate to get PMF, move to CPU

# Print theoretical and empirical results
print(f"Theoretical Expected Value E[X] = {expected_value:.2f}")
print(f"Theoretical Variance Var(X) = {variance:.2f}")
print(f"Theoretical Standard Deviation = {std_dev:.2f}")
print(f"Empirical Mean = {empirical_mean:.2f}")
print(f"Empirical Variance = {empirical_variance:.2f}")

# Plot PMF with expected value and variance bounds
plt.stem(x.cpu().numpy(), pmf, basefmt=" ", linefmt='b-', markerfmt='bo', label='PMF')
plt.axvline(x=expected_value, color='r', linestyle='--', label=f'E[X] = {expected_value:.2f}')
plt.axvline(x=expected_value - std_dev, color='g', linestyle=':', label=f'μ ± σ')
plt.axvline(x=expected_value + std_dev, color='g', linestyle=':')
plt.title(f"Binomial Distribution (n={n}, p={p})")
plt.xlabel("Number of Successes (k)")
plt.ylabel("PMF")
plt.xticks(np.arange(0, n + 1, 2))  # X-axis ticks every 2
plt.ylim(0, max(pmf) * 1.2)  # Extend y-axis for visibility
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
