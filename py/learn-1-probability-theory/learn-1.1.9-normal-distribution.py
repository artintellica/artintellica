import torch
import matplotlib.pyplot as plt
import numpy as np

# Set device to MPS if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Normal distribution parameters
mu = 0.0  # Mean
sigma = 1.0  # Standard deviation
normal_dist = torch.distributions.Normal(loc=torch.tensor(mu, device=device), 
                                        scale=torch.tensor(sigma, device=device))

# Values to evaluate PDF (x from -4 to 4)
x = torch.linspace(-4, 4, 1000, device=device)

# Compute PDF using log_prob for numerical stability
log_pdf = normal_dist.log_prob(x)
pdf = torch.exp(log_pdf).cpu().numpy()  # Exponentiate to get PDF, move to CPU for plotting

# Print PDF at a few points
print(f"PDF for Normal (μ={mu}, σ={sigma}) at selected points:")
for i, val in enumerate([-2, 0, 2]):
    x_val = x[int(250 * (i + 1))].item()  # Approximate -2, 0, 2
    print(f"f(x={x_val:.2f}) = {pdf[int(250 * (i + 1))]:.4f}")

# Plot PDF as a line plot with filled area
plt.plot(x.cpu().numpy(), pdf, 'm-', linewidth=2, label='PDF')
plt.fill_between(x.cpu().numpy(), pdf, alpha=0.2, color='magenta')
plt.title(f"Normal Distribution (μ={mu}, σ={sigma})")
plt.xlabel("x")
plt.ylabel("PDF")
plt.ylim(0, 0.5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
