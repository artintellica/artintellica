import torch
import matplotlib.pyplot as plt
import numpy as np

# Set device to MPS (GPU) if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Bivariate normal distribution parameters
mean = torch.tensor([0.0, 0.0], device=device)  # Mean vector [mu_1, mu_2]
covariance_matrix = torch.tensor([[1.0, 0.5], [0.5, 1.0]], device=device)  # Covariance matrix
dist = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=covariance_matrix)

# Generate samples
n_samples = 1000  # Number of samples
samples = dist.sample((n_samples,)).to(device)  # Shape: (n_samples, 2)

# Compute empirical covariance matrix
empirical_mean = samples.mean(dim=0)  # Mean of each dimension
centered_samples = samples - empirical_mean  # Center the samples
empirical_cov = torch.mm(centered_samples.t(), centered_samples) / (n_samples - 1)  # Covariance matrix

# Generate grid for contour plot of PDF
x = torch.linspace(-3, 3, 100, device=device)
y = torch.linspace(-3, 3, 100, device=device)
X, Y = torch.meshgrid(x, y, indexing='ij')
grid = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)  # Shape: (10000, 2)
log_prob = dist.log_prob(grid).reshape(100, 100)  # Log-PDF
pdf = torch.exp(log_prob).cpu().numpy()  # PDF for contour plot

# Print theoretical and empirical results
print(f"Theoretical Mean: {mean.cpu().numpy()}")
print(f"Theoretical Covariance Matrix:\n{covariance_matrix.cpu().numpy()}")
print(f"Empirical Mean: {empirical_mean.cpu().numpy()}")
print(f"Empirical Covariance Matrix:\n{empirical_cov.cpu().numpy()}")

# Plot scatter and contour
plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0].cpu().numpy(), samples[:, 1].cpu().numpy(), alpha=0.5, s=10, label='Samples')
plt.contour(X.cpu().numpy(), Y.cpu().numpy(), pdf, levels=5, colors='red', label='PDF Contours')
plt.title("Bivariate Normal Distribution (μ=[0,0], Σ=[[1,0.5],[0.5,1]])")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis('equal')  # Equal scaling for better visualization
plt.show()
