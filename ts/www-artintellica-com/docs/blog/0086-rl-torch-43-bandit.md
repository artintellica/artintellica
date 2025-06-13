+++
title = "Learn Reinforcement Learning with PyTorch, Part 4.3: Bandit Problems—Exploration vs. Exploitation"
author = "Artintellica"
date = "2025-06-13"
+++

## Introduction

One of the most iconic settings in reinforcement learning is the **multi-armed bandit** problem. Here, you repeatedly choose between several noisy options (“arms”), each with an unknown reward distribution. The catch? You must **balance exploration** (learning about arms) and **exploitation** (favoring what works best so far).

Bandits are the simplest setting for exploring “exploration vs. exploitation”—a core problem in RL and real-world decision-making (ads, A/B tests, drug trials, online recommendations…).

In this post, you’ll:

- Build and visualize a multi-armed bandit environment.
- Implement and compare classic exploration strategies: epsilon-greedy, UCB, and Thompson Sampling.
- Track arm selections, rewards, and learning curves.
- See all code in clear, reproducible PyTorch/Numpy demos.

---

## Mathematics: Bandit Setting and Exploration Strategies

**Multi-armed Bandit:**  
- $K$ arms, each with an unknown (often stationary) reward distribution $R_k$.
- At each time $t$, you pick arm $a_t \in \{1,...,K\}$, and observe random reward $r_t \sim R_{a_t}$.

**Goal:** Maximize expected total reward; equivalently, minimize **regret** versus always picking the best arm.

### **Epsilon-Greedy Policy**

- With probability $\varepsilon$, pick an arm *at random* (**explore**).
- Otherwise, pick the arm with highest average observed reward (**exploit**).

### **Upper Confidence Bound (UCB)**

For each arm $k$:
\[
Q_k(t) + c \sqrt{\frac{\ln t}{N_k(t)}}
\]
- Exploit arms with higher estimates $Q_k(t)$, but also explore arms with fewer pulls $N_k(t)$.

### **Thompson Sampling**

- Maintain a belief over the parameters of the reward distribution for each arm.
- Sample from each belief, pick the arm with the highest sampled value.
- E.g., for Bernoulli rewards, use a Beta posterior: Beta($\alpha_k$, $\beta_k$).

---

## Explanation: How the Math Connects to Code

- **Reward Distributions:** For each arm, define a true but hidden mean and sample actual rewards each pull.
- **Agent Estimates:** Track per-arm counts and average rewards.
- **Policies:**
    - *Epsilon-greedy:* select random vs. best.
    - *UCB:* use a confidence upper bound to bias towards least tried arms.
    - *Thompson Sampling:* update a Beta prior after each observation, then sample from this prior.
- **Metrics/Visualization:** Plot histograms of arm means, arm selection frequencies, average reward over time (the “learning curve”).

---

## Python Demonstrations

### Demo 1: Simulate a Multi-Armed Bandit and Plot Arm Payout Distributions

```python
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
            ys = 1/np.sqrt(2*np.pi*0.1**2) * np.exp(-(xs-mu)**2/(2*0.1**2))
            plt.plot(xs, ys, label=f"Arm {k} (mean={mu:.2f})")
        plt.legend()
        plt.title("Bandit Arm Reward Distributions")
        plt.ylabel("Density")
        plt.xlabel("Reward")
        plt.show()

env = BanditEnv(K=5)
print("True arm means:", np.round(env.true_means, 2))
env.plot_distributions()
```

---

### Demo 2: Implement Epsilon-Greedy Policy and Track Arm Selections

```python
np.random.seed(17)

K = 5
env = BanditEnv(K)
n_steps = 300
eps = 0.1

counts = np.zeros(K, dtype=int)         # Number of times each arm was pulled
values = np.zeros(K)                    # Empirical mean reward for each arm
selections = np.zeros(K, dtype=int)     # For stats

rewards = []

for t in range(1, n_steps+1):
    if np.random.rand() < eps:
        arm = np.random.choice(K)
    else:
        arm = np.argmax(values)
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
```

---

### Demo 3: Compare UCB and Thompson Sampling on the Same Problem

#### UCB

```python
def bandit_ucb(env: BanditEnv, n_steps: int = 300, c: float = 1.0) -> tuple[list[float], np.ndarray]:
    K = env.K
    counts = np.zeros(K)
    values = np.zeros(K)
    rewards = []
    selections = np.zeros(K, dtype=int)

    for t in range(1, n_steps+1):
        # Pull each arm once to start
        if t <= K:
            arm = t-1
        else:
            ucb = values + c * np.sqrt(np.log(t) / (counts + 1e-7))
            arm = np.argmax(ucb)
        reward = env.pull(arm)
        counts[arm] += 1
        selections[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        rewards.append(reward)
    return rewards, selections

rewards_ucb, selections_ucb = bandit_ucb(env, n_steps)
plt.bar(range(K), selections_ucb)
plt.xlabel("Arm"); plt.ylabel("# chosen")
plt.title("UCB Selection")
plt.show()

plt.plot(np.cumsum(rewards_ucb) / (np.arange(n_steps) + 1), label="UCB")
```

#### Thompson Sampling (for Bernoulli Arms)

For demonstration, let’s use Bernoulli arms:

```python
class BernoulliBanditEnv:
    def __init__(self, K: int = 5) -> None:
        self.K = K
        self.true_means = np.random.uniform(0.1, 0.9, K)
    def pull(self, arm: int) -> int:
        return int(np.random.rand() < self.true_means[arm])
envb = BernoulliBanditEnv(K=5)

def bandit_thompson(env, n_steps=300) -> tuple[list[float], np.ndarray]:
    K = env.K
    alpha = np.ones(K)      # Successes + 1
    beta = np.ones(K)       # Failures + 1
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
plt.xlabel("Arm"); plt.ylabel("# chosen")
plt.title("Thompson Sampling Selection")
plt.show()

plt.plot(np.cumsum(rewards_th)/ (np.arange(len(rewards_th)) + 1), label="Thompson")
plt.title("Average Reward Over Time")
plt.xlabel("Step"); plt.ylabel("Average Reward")
plt.legend(); plt.show()
```

---

### Demo 4: Plot Cumulative Reward for Different Strategies

Let's show learning curves side-by-side:

```python
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

plt.plot(np.cumsum(rewards_eps)/ (np.arange(n_steps) + 1), label="Epsilon-Greedy")
plt.plot(np.cumsum(rewards_ucb)/ (np.arange(n_steps) + 1), label="UCB")
plt.plot(np.cumsum(rewards_th)/ (np.arange(n_steps) + 1), label="Thompson")
plt.title("Cumulative Average Reward: Bandit Algorithms")
plt.xlabel("Step"); plt.ylabel("Avg Reward")
plt.legend(); plt.grid(True); plt.show()
```

#### **Project Exercise:** Visualize the learning curve of a bandit agent over time  
The plot above is exactly this!

---

## Exercises

### **Exercise 1:** Simulate a Multi-Armed Bandit and Plot Arm Payout Distributions

- Create a bandit environment with at least three arms with random means.
- Plot the true reward distributions of each arm.

---

### **Exercise 2:** Implement Epsilon-Greedy Policy and Track Arm Selections

- Simulate a single agent for 200+ steps using epsilon-greedy.
- Count how often each arm is selected.
- Plot selection histogram and average reward curve.

---

### **Exercise 3:** Compare UCB and Thompson Sampling on the Same Problem

- Implement both methods for the same bandit instance.
- Track arm selections and cumulative rewards.
- Plot results side-by-side.

---

### **Exercise 4:** Plot Cumulative Reward for Different Strategies

- Compare epsilon-greedy, UCB, and Thompson sampling.
- Plot all learning curves on a single chart for direct comparison.

---

### **Sample Starter Code for Exercises**

```python
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
    ys = 1/np.sqrt(2*np.pi*0.1**2) * np.exp(-(xs-mu)**2/(2*0.1**2))
    plt.plot(xs, ys, label=f"Arm {k}")
plt.legend(); plt.title("Arm Reward Densities"); plt.show()

# Epsilon-greedy agent
K = env.K
counts = np.zeros(K)
values = np.zeros(K)
rewards = []
for t in range(250):
    arm = np.random.choice(K) if np.random.rand() < 0.1 else np.argmax(values)
    reward = env.pull(arm)
    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]
    rewards.append(reward)
plt.bar(range(K), counts); plt.title("Arm Selections"); plt.show()
plt.plot(np.cumsum(rewards)/(np.arange(len(rewards))+1))
plt.title("Average Reward"); plt.show()

# (Re-use demos above for UCB, Thompson, and final comparison.)
```

---

## Conclusion

You’ve now seen the theory and practice of bandit problems and main exploration algorithms—*epsilon-greedy*, *UCB*, and *Thompson Sampling*—and how to empirically compare them. These tools are not just educational; they’re deployed in real online systems and form the intuition behind RL’s exploration dilemmas.

Next up: value-based RL methods—where the agent can *plan ahead* and learn from future, not just immediate, reward!

See you in Part 4.4!
