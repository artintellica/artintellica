import torch
import matplotlib.pyplot as plt
import numpy as np

# Set device to MPS (GPU) if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Bernoulli distribution parameters (fair coin)
p = 0.5  # Probability of heads
bernoulli_dist = torch.distributions.Bernoulli(probs=torch.tensor(p, device=device))

# Parameters for simulation
n_trials = 10000  # Maximum number of trials
n_simulations = 100  # Number of independent sequences for almost sure convergence
epsilon = 0.1  # Threshold for convergence in probability

# Simulate data for convergence in probability
samples = bernoulli_dist.sample((n_simulations, n_trials)).to(device)  # Shape: (n_simulations, n_trials)
cumsum = torch.cumsum(samples, dim=1)  # Cumulative sum along trials
n = torch.arange(1, n_trials + 1, device=device, dtype=torch.float32)  # Trial numbers
sample_means = cumsum / n  # Sample mean for each simulation and trial

# Compute P(|sample_mean - mu| > epsilon) for convergence in probability
mu = p  # Expected value
prob_deviation = torch.mean((torch.abs(sample_means - mu) > epsilon).float(), dim=0)  # Average over simulations

# Plot results
plt.figure(figsize=(12, 5))

# Subplot 1: Convergence in Probability
plt.subplot(1, 2, 1)
plt.plot(n.cpu().numpy(), prob_deviation.cpu().numpy(), 'b-', label=f'P(|mean - {mu}| > {epsilon})')
plt.axhline(y=0, color='r', linestyle='--', label='Target (0)')
plt.title(f"Convergence in Probability (ε={epsilon})")
plt.xlabel("Number of Trials (n)")
plt.ylabel("Probability")
plt.xscale('log')  # Log scale for x-axis
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 2: Almost Sure Convergence
plt.subplot(1, 2, 2)
for i in range(5):  # Plot 5 trajectories for clarity
    plt.plot(n.cpu().numpy(), sample_means[i].cpu().numpy(), alpha=0.5, label=f'Trajectory {i+1}' if i < 2 else None)
plt.axhline(y=mu, color='r', linestyle='--', label=f'E[X] = {mu}')
plt.title("Almost Sure Convergence")
plt.xlabel("Number of Trials (n)")
plt.ylabel("Sample Mean")
plt.xscale('log')  # Log scale for x-axis
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Print final probability for convergence in probability
print(f"Theoretical Expected Value E[X] = {mu:.2f}")
print(f"Final P(|mean - {mu}| > {epsilon}) after {n_trials} trials = {prob_deviation[-1].item():.4f}")
