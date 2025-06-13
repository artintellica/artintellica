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


class BernoulliBanditEnv:
    def __init__(self, K: int = 5) -> None:
        self.K = K
        self.true_means = np.random.uniform(0.1, 0.9, K)

    def pull(self, arm: int) -> int:
        return int(np.random.rand() < self.true_means[arm])


def bandit_ucb(
    env: BanditEnv | BernoulliBanditEnv, n_steps: int = 300, c: float = 1.0
) -> tuple[list[float], np.ndarray]:
    K = env.K
    counts = np.zeros(K)
    values = np.zeros(K)
    rewards = []
    selections = np.zeros(K, dtype=int)

    for t in range(1, n_steps + 1):
        # Pull each arm once to start
        if t <= K:
            arm = t - 1
        else:
            ucb = values + c * np.sqrt(np.log(t) / (counts + 1e-7))
            arm = np.argmax(ucb)
        arm = cast(int, arm)
        reward = env.pull(arm)
        counts[arm] += 1
        selections[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        rewards.append(reward)
    return rewards, selections


rewards_ucb, selections_ucb = bandit_ucb(env, n_steps)
plt.bar(range(K), selections_ucb)
plt.xlabel("Arm")
plt.ylabel("# chosen")
plt.title("UCB Selection")
plt.show()

plt.plot(np.cumsum(rewards_ucb) / (np.arange(n_steps) + 1), label="UCB")
envb = BernoulliBanditEnv(K=5)


def bandit_thompson(env, n_steps=300) -> tuple[list[float], np.ndarray]:
    K = env.K
    alpha = np.ones(K)  # Successes + 1
    beta = np.ones(K)  # Failures + 1
    rewards = []
    selections = np.zeros(K, dtype=int)

    for t in range(n_steps):
        samples = np.random.beta(alpha, beta)
        arm = np.argmax(samples)
        reward = env.pull(arm)
        selections[arm] += 1
        if reward:
            alpha[arm] += 1
        else:
            beta[arm] += 1
        rewards.append(reward)
    return rewards, selections


rewards_th, selections_th = bandit_thompson(envb)
plt.bar(range(K), selections_th)
plt.xlabel("Arm")
plt.ylabel("# chosen")
plt.title("Thompson Sampling Selection")
plt.show()

plt.plot(np.cumsum(rewards_th) / (np.arange(len(rewards_th)) + 1), label="Thompson")
plt.title("Average Reward Over Time")
plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.legend()
plt.show()

# Use the same Bernoulli env for all methods for fair comparison
envb2 = BernoulliBanditEnv(K=5)
n_steps = 300


def bandit_eps_greedy(env, n_steps=300, eps=0.1):
    K = env.K
    counts = np.zeros(K)
    values = np.zeros(K)
    rewards = []
    for t in range(n_steps):
        if np.random.rand() < eps:
            arm = np.random.choice(K)
        else:
            arm = np.argmax(values)
        reward = env.pull(arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        rewards.append(reward)
    return rewards


rewards_eps = bandit_eps_greedy(envb2, n_steps=n_steps, eps=0.1)
rewards_ucb, _ = bandit_ucb(envb2, n_steps=n_steps, c=1.2)
rewards_th, _ = bandit_thompson(envb2, n_steps=n_steps)

plt.plot(np.cumsum(rewards_eps) / (np.arange(n_steps) + 1), label="Epsilon-Greedy")
plt.plot(np.cumsum(rewards_ucb) / (np.arange(n_steps) + 1), label="UCB")
plt.plot(np.cumsum(rewards_th) / (np.arange(n_steps) + 1), label="Thompson")
plt.title("Cumulative Average Reward: Bandit Algorithms")
plt.xlabel("Step")
plt.ylabel("Avg Reward")
plt.legend()
plt.grid(True)
plt.show()
