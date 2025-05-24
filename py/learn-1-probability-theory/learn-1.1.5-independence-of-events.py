import torch
import matplotlib.pyplot as plt

# Set device to MPS if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Simulation parameters
n_rolls = 100000  # Number of trials

# Simulate two dice rolls (values 1 to 6)
dice1 = torch.randint(low=1, high=7, size=(n_rolls,), device=device)
dice2 = torch.randint(low=1, high=7, size=(n_rolls,), device=device)

# Event A: First die is 4
event_a = (dice1 == 4)
p_a = event_a.float().mean().item()  # P(A)

# Event B: Second die is odd (1, 3, or 5)
event_b = (dice2 % 2 == 1)
p_b = event_b.float().mean().item()  # P(B)

# Joint event A and B
joint_event = event_a & event_b
p_a_and_b = joint_event.float().mean().item()  # P(A ∩ B)

# Theoretical joint probability for independent events
p_a_times_p_b = p_a * p_b

# Print results
print(f"P(A) = {p_a:.4f} (First die is 4)")
print(f"P(B) = {p_b:.4f} (Second die is odd)")
print(f"P(A ∩ B) = {p_a_and_b:.4f} (Empirical)")
print(f"P(A) * P(B) = {p_a_times_p_b:.4f} (Theoretical)")
print(f"Difference |P(A ∩ B) - P(A)P(B)| = {abs(p_a_and_b - p_a_times_p_b):.6f}")

# Visualize probabilities
labels = ['P(A)', 'P(B)', 'P(A ∩ B)', 'P(A)P(B)']
values = [p_a, p_b, p_a_and_b, p_a_times_p_b]
plt.bar(labels, values, color=['blue', 'green', 'red', 'purple'])
plt.title("Probabilities for Dice Roll Events")
plt.ylabel("Probability")
plt.ylim(0, 0.6)
plt.show()
