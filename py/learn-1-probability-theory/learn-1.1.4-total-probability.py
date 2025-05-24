import torch
import matplotlib.pyplot as plt

# Set device to MPS if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Probabilities of the partition
p_b = torch.tensor([0.01, 0.04, 0.95], device=device)  # P(B1), P(B2), P(B3)

# Conditional probabilities P(A|Bi)
p_a_given_b = torch.tensor([0.99, 0.10, 0.02], device=device)  # P(A|B1), P(A|B2), P(A|B3)

# Law of Total Probability: P(A) = sum(P(A|Bi) * P(Bi))
p_a = torch.sum(p_a_given_b * p_b)

print(f"P(A) = {p_a.item():.4f}")

# Visualize contributions
labels = ['Disease', 'Related Condition', 'Healthy']
contributions = (p_a_given_b * p_b).cpu().numpy()  # Move to CPU for plotting
plt.bar(labels, contributions)
plt.title("Contributions to P(Test Positive)")
plt.ylabel("Probability Contribution")
plt.show()
