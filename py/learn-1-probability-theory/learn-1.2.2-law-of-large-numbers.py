import torch
import matplotlib.pyplot as plt
import numpy as np

# Set device to MPS (GPU) if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Bernoulli distribution parameters (fair coin)
p = 0.5  # Probability of heads (success)
bernoulli_dist = torch.distributions.Bernoulli(probs=torch.tensor(p, device=device))

# Simulate coin flips
n_trials = 10000  # Number of coin flips
samples = bernoulli_dist.sample((n_trials,)).to(device)  # 1 for heads, 0 for tails

# Compute running sample mean (cumulative average)
cumsum = torch.cumsum(samples, dim=0)  # Cumulative sum of heads
n = torch.arange(1, n_trials + 1, device=device, dtype=torch.float32)  # Trial numbers
running_mean = cumsum / n  # Running mean at each trial

# Theoretical expected value
expected_value = p  # E[X] = p for Bernoulli

# Print final empirical mean
print(f"Theoretical Expected Value E[X] = {expected_value:.2f}")
print(f"Empirical Mean after {n_trials} trials = {running_mean[-1].item():.4f}")

# Plot running mean with theoretical expected value
plt.plot(n.cpu().numpy(), running_mean.cpu().numpy(), 'b-', label='Running Mean')
plt.axhline(y=expected_value, color='r', linestyle='--', label=f'E[X] = {expected_value:.2f}')
plt.title("Law of Large Numbers: Bernoulli (p=0.5)")
plt.xlabel("Number of Trials")
plt.ylabel("Sample Mean")
plt.xscale('log')  # Log scale for x-axis to emphasize early trials
plt.ylim(0, 1)  # Y-axis range for clarity
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
