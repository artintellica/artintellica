import numpy as np
import matplotlib.pyplot as plt


class BanditEnv:
    def __init__(self, K: int = 4) -> None:
        self.K = K
        self.true_means = np.random.uniform(0.2, 0.9, K)

    def pull(self, arm: int) -> float:
        return np.random.normal(self.true_means[arm], 0.1)


env = BanditEnv()
print("True means:", env.true_means)
for k, mu in enumerate(env.true_means):
    xs = np.linspace(0, 1.5, 80)
    ys = 1 / np.sqrt(2 * np.pi * 0.1**2) * np.exp(-((xs - mu) ** 2) / (2 * 0.1**2))
    plt.plot(xs, ys, label=f"Arm {k}")
plt.legend()
plt.title("Arm Reward Densities")
plt.show()

# Epsilon-greedy agent
K = env.K
counts = np.zeros(K)
values = np.zeros(K)
rewards = []
for t in range(250):
    arm = np.random.choice(K) if np.random.rand() < 0.1 else np.argmax(values)
    arm = int(arm)  # Ensure arm is an integer index
    reward = env.pull(arm)
    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]
    rewards.append(reward)
plt.bar(range(K), counts)
plt.title("Arm Selections")
plt.show()
plt.plot(np.cumsum(rewards) / (np.arange(len(rewards)) + 1))
plt.title("Average Reward")
plt.show()

# (Re-use demos above for UCB, Thompson, and final comparison.)
