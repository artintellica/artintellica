import numpy as np
import matplotlib.pyplot as plt

def plot_state_growth(base: int = 10, max_vars: int = 12) -> None:
    num_vars = np.arange(1, max_vars+1)
    num_states = base ** num_vars
    plt.figure(figsize=(7,4))
    plt.semilogy(num_vars, num_states, marker='o')
    plt.axhline(1_000_000, color='red', linestyle='--', label="1 million states")
    plt.title(f"Exponential State Space Growth (base={base})")
    plt.xlabel("Number of state variables")
    plt.ylabel("Number of possible states (log scale)")
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.show()

plot_state_growth()
