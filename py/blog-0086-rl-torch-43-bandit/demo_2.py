import numpy as np
import matplotlib.pyplot as plt
from typing import cast


class BanditEnv:
    def __init__(self, K: int = 5) -> None:
        self.K = K
        self.true_means = np.random.uniform(0, 1, K)

    def pull(self, arm: int) -> float:
        # Rewards are Gaussian around each mean
        return np.random.normal(self.true_means[arm], 0.1)

    def plot_distributions(self) -> None:
        for k, mu in enumerate(self.true_means):
            xs = np.linspace(0, 1.5, 100)
            ys = (
                1
                / np.sqrt(2 * np.pi * 0.1**2)
                * np.exp(-((xs - mu) ** 2) / (2 * 0.1**2))
            )
            plt.plot(xs, ys, label=f"Arm {k} (mean={mu:.2f})")
        plt.legend()
        plt.title("Bandit Arm Reward Distributions")
        plt.ylabel("Density")
        plt.xlabel("Reward")
        plt.show()


env = BanditEnv(K=5)
print("True arm means:", np.round(env.true_means, 2))
# env.plot_distributions()

np.random.seed(17)

K = 5
env = BanditEnv(K)
n_steps = 300
eps = 0.1

counts = np.zeros(K, dtype=int)  # Number of times each arm was pulled
values = np.zeros(K)  # Empirical mean reward for each arm
selections = np.zeros(K, dtype=int)  # For stats

rewards = []

for t in range(1, n_steps + 1):
    if np.random.rand() < eps:
        arm = np.random.choice(K)
    else:
        arm = np.argmax(values)
    arm = cast(int, arm)  # Ensure arm is an integer index
    reward = env.pull(arm)
    counts[arm] += 1
    selections[arm] += 1
    # Update running mean for empirical value
    values[arm] += (reward - values[arm]) / counts[arm]
    rewards.append(reward)

plt.bar(range(K), selections)
plt.xlabel("Arm")
plt.ylabel("# times selected")
plt.title(f"Arm Selection Counts (epsilon={eps})")
plt.show()

plt.plot(np.cumsum(rewards) / (np.arange(n_steps) + 1))
plt.title("Epsilon-Greedy: Average Reward Over Time")
plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.show()
