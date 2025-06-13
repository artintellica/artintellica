import numpy as np
import matplotlib.pyplot as plt


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
env.plot_distributions()
