import torch
import matplotlib.pyplot as plt
import numpy as np

# Set device to MPS (GPU) if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define two Bernoulli distributions
p = 0.3  # Probability for distribution P
q = 0.5  # Probability for distribution Q
dist_p = torch.distributions.Bernoulli(probs=torch.tensor(p, device=device))
dist_q = torch.distributions.Bernoulli(probs=torch.tensor(q, device=device))

# Values to evaluate PMF (x = 0, 1)
x = torch.tensor([0.0, 1.0], device=device)

# Compute PMFs
pmf_p = torch.exp(dist_p.log_prob(x)).cpu().numpy()  # P(X=x)
pmf_q = torch.exp(dist_q.log_prob(x)).cpu().numpy()  # Q(X=x)

# Compute entropy: H(X) = -sum(p_i * log(p_i))
entropy_p = -torch.sum(dist_p.log_prob(x) * torch.exp(dist_p.log_prob(x))).item()
entropy_q = -torch.sum(dist_q.log_prob(x) * torch.exp(dist_q.log_prob(x))).item()

# Compute KL Divergence: D_KL(P || Q) = sum(p_i * log(p_i / q_i))
kl_divergence = torch.sum(
    torch.exp(dist_p.log_prob(x)) * (dist_p.log_prob(x) - dist_q.log_prob(x))
).item()

# Print results
print(f"Entropy of P (p={p}): {entropy_p:.4f} bits")
print(f"Entropy of Q (q={q}): {entropy_q:.4f} bits")
print(f"KL Divergence D_KL(P || Q): {kl_divergence:.4f} bits")

# Plot PMFs
plt.figure(figsize=(8, 6))
width = 0.35
plt.bar(
    x.cpu().numpy() - width / 2,
    pmf_p,
    width,
    label=f"P (p={p})",
    color="blue",
    alpha=0.5,
)
plt.bar(
    x.cpu().numpy() + width / 2,
    pmf_q,
    width,
    label=f"Q (q={q})",
    color="red",
    alpha=0.5,
)
plt.title("Bernoulli PMFs and Information Theory Metrics")
plt.xlabel("x")
plt.ylabel("Probability")
plt.xticks([0, 1])
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(
    0.5,
    0.9,
    f"H(P) = {entropy_p:.4f} bits\nH(Q) = {entropy_q:.4f} bits\nD_KL(P || Q) = {kl_divergence:.4f} bits",
    transform=plt.gca().transAxes,
    ha="right",
    va="top",
    bbox=dict(facecolor="white", alpha=0.8),
)
plt.show()
