import torch
import matplotlib.pyplot as plt
import numpy as np

# Set device to MPS (GPU) if available, else CPU
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

# Poisson distribution parameters
lam = 3.0  # Rate parameter (lambda)
poisson_dist = torch.distributions.Poisson(rate=torch.tensor(lam, device=device))

# Theoretical mean and variance
mu = lam  # E[X] = lambda
sigma2 = lam  # Var(X) = lambda
sigma = np.sqrt(sigma2)  # Standard deviation

# Define range for threshold values a
a_values = torch.linspace(1, 10, 50, device=device)  # Thresholds for P(X >= a) and P(|X - mu| >= a)

# Compute Markov's Inequality bound: P(X >= a) <= E[X] / a
markov_bound = mu / a_values

# Compute Chebyshev's Inequality bound: P(|X - mu| >= a) <= sigma^2 / a^2
chebyshev_bound = sigma2 / (a_values ** 2)

# Estimate empirical probabilities using samples
n_samples = 100000  # Number of samples
samples = poisson_dist.sample((n_samples,)).to(device)
empirical_markov = torch.zeros_like(a_values)
empirical_chebyshev = torch.zeros_like(a_values)
for i, a in enumerate(a_values):
    empirical_markov[i] = (samples >= a).float().mean()  # P(X >= a)
    empirical_chebyshev[i] = ((samples - mu).abs() >= a).float().mean()  # P(|X - mu| >= a)

# Print results for a specific a (e.g., a = 5)
a_example = 5.0
print(f"For a = {a_example}:")
print(f"Markov's Bound: P(X >= {a_example}) <= {mu / a_example:.4f}")
print(f"Empirical P(X >= {a_example}) = {empirical_markov[int(len(a_values) * 4 / 10)].item():.4f}")
print(f"Chebyshev's Bound: P(|X - mu| >= {a_example}) <= {sigma2 / (a_example ** 2):.4f}")
print(f"Empirical P(|X - mu| >= {a_example}) = {empirical_chebyshev[int(len(a_values) * 4 / 10)].item():.4f}")

# Plot bounds and empirical probabilities
plt.figure(figsize=(10, 6))
plt.plot(a_values.cpu().numpy(), markov_bound.cpu().numpy(), 'b-', label="Markov's Bound")
plt.plot(a_values.cpu().numpy(), empirical_markov.cpu().numpy(), 'b--', label="Empirical P(X >= a)")
plt.plot(a_values.cpu().numpy(), chebyshev_bound.cpu().numpy(), 'r-', label="Chebyshev's Bound")
plt.plot(a_values.cpu().numpy(), empirical_chebyshev.cpu().numpy(), 'r--', label="Empirical P(|X - mu| >= a)")
plt.title(f"Markov’s and Chebyshev’s Inequalities: Poisson (λ={lam})")
plt.xlabel("Threshold a")
plt.ylabel("Probability")
plt.yscale('log')  # Log scale for y-axis to show small probabilities
plt.ylim(1e-4, 1)  # Adjust y-axis
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
