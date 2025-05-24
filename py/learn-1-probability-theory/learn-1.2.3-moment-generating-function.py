import torch
import matplotlib.pyplot as plt
import numpy as np

# Set device to MPS (GPU) if available, else CPU
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

# Poisson distribution parameters
lam = 2.0  # Rate parameter (lambda)
poisson_dist = torch.distributions.Poisson(rate=torch.tensor(lam, device=device))

# Define range for t values to evaluate MGF
t_values = torch.linspace(-0.5, 0.5, 100, device=device)  # Small range where MGF exists

# Compute theoretical MGF: M(t) = exp(lambda * (exp(t) - 1))
theoretical_mgf = torch.exp(lam * (torch.exp(t_values) - 1))

# Compute empirical MGF using samples
n_samples = 100000  # Number of samples
samples = poisson_dist.sample((n_samples,)).to(device)
empirical_mgf = torch.zeros_like(t_values)
for i, t in enumerate(t_values):
    empirical_mgf[i] = torch.mean(torch.exp(t * samples))  # E[exp(tX)] approximated

# Derive moments from theoretical MGF
# First moment: E[X] = M'(0) = lambda
# Second moment: E[X^2] = M''(0) = lambda + lambda^2
# Variance: Var(X) = E[X^2] - (E[X])^2 = lambda
moment1 = lam  # E[X]
moment2 = lam + lam**2  # E[X^2]
variance = moment2 - moment1**2  # Var(X)

# Empirical moments from samples
empirical_mean = samples.mean().item()
empirical_variance = samples.var().item()

# Print results
print(f"Theoretical MGF: M(t) = exp({lam} * (exp(t) - 1))")
print(f"Theoretical E[X] = {moment1:.2f}")
print(f"Theoretical Var(X) = {variance:.2f}")
print(f"Empirical Mean = {empirical_mean:.2f}")
print(f"Empirical Variance = {empirical_variance:.2f}")

# Plot theoretical and empirical MGF
plt.plot(t_values.cpu().numpy(), theoretical_mgf.cpu().numpy(), 'b-', label='Theoretical MGF')
plt.plot(t_values.cpu().numpy(), empirical_mgf.cpu().numpy(), 'r--', label='Empirical MGF')
plt.title(f"Moment-Generating Function: Poisson (λ={lam})")
plt.xlabel("t")
plt.ylabel("M(t)")
plt.ylim(0, theoretical_mgf.max().item() * 1.2)  # Adjust y-axis
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
