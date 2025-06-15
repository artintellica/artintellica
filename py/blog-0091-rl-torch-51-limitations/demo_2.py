import matplotlib.pyplot as plt

def plot_state_space_growth(k: int, max_vars: int) -> None:
    """
    Plot number of possible states vs number of state variables for discrete variables.
    """
    import numpy as np
    num_vars = list(range(1, max_vars+1))
    num_states = [k ** n for n in num_vars]
    plt.figure(figsize=(7,4))
    plt.semilogy(num_vars, num_states, marker='o')
    plt.title(f"Exponential Growth of State Space (k={k})")
    plt.xlabel("Number of state variables (dimensions)")
    plt.ylabel("Number of possible states (log scale)")
    plt.grid(True, which='both')
    plt.show()

plot_state_space_growth(k=10, max_vars=10)
