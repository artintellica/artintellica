import numpy as np


# Function to calculate target dimension k based on Johnson-Lindenstrauss lemma
def calculate_jl_dimension(n, epsilon):
    """
    Calculate the target dimension k for random projection using the Johnson-Lindenstrauss lemma.

    Parameters:
    n (int): Number of samples (points)
    epsilon (float): Distortion factor (0 < epsilon < 1)

    Returns:
    int: Target dimension k = (8 * log(n)) / epsilon^2
    """
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("Epsilon must be between 0 and 1")
    if n < 1:
        raise ValueError("Number of samples n must be positive")

    k = (8 * np.log(n)) / (epsilon**2)
    return int(k)


# Test the function for specified values of n and epsilon
n_values = [100, 1000, 10000]
epsilon_values = [0.1, 0.2]

# Print results in a formatted way
print(
    "Target Dimension k based on Johnson-Lindenstrauss Lemma (k = 8 * log(n) / epsilon^2)"
)
print("-" * 70)
print(f"{'n (samples)':<15} {'epsilon':<10} {'k (target dimension)':<20}")
print("-" * 70)

for n in n_values:
    for eps in epsilon_values:
        k = calculate_jl_dimension(n, eps)
        print(f"{n:<15} {eps:<10.1f} {k:<20}")
