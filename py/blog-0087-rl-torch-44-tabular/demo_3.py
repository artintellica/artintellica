import numpy as np

n_states = 4
n_actions = 2
Q = np.zeros((n_states, n_actions))  # Q[state, action]

print("Initial Q-table:")
print(Q)

import random


class SimpleGridEnv:
    def __init__(self, n=4):
        self.n = n
        self.reset()

    def reset(self) -> int:
        self.s = 0
        return self.s

    def step(self, a: int) -> tuple[int, float, bool]:
        # a: 0=LEFT, 1=RIGHT
        if a == 1:
            self.s += 1
        else:
            self.s -= 1
        self.s = np.clip(self.s, 0, self.n - 1)
        reward = 1.0 if self.s == self.n - 1 else 0.0
        done = self.s == self.n - 1
        return self.s, reward, done


def epsilon_greedy(Q, s, eps=0.2):
    if random.random() < eps:
        return random.choice(range(n_actions))
    return int(np.argmax(Q[s]))


def train_q(env, Q, episodes=250, alpha=0.1, gamma=0.9, eps=0.2):
    returns = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            a = epsilon_greedy(Q, s, eps)
            s2, r, done = env.step(a)
            Q[s, a] += alpha * (
                r + gamma * np.max(Q[s2]) - Q[s, a]
            )  # Q-learning update
            s = s2
            total += r
        returns.append(total)
    return np.array(returns)


def train_sarsa(env, Q, episodes=250, alpha=0.1, gamma=0.9, eps=0.2):
    returns = []
    for ep in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, eps)
        done = False
        total = 0.0
        while not done:
            s2, r, done = env.step(a)
            a2 = epsilon_greedy(Q, s2, eps)
            Q[s, a] += alpha * (r + gamma * Q[s2, a2] - Q[s, a])  # SARSA update
            s = s2
            a = a2
            total += r
        returns.append(total)
    return np.array(returns)


env = SimpleGridEnv(n_states)
Q1 = np.zeros((n_states, n_actions))
Q2 = np.zeros((n_states, n_actions))
rets1 = train_q(env, Q1)
rets2 = train_sarsa(env, Q2)

import matplotlib.pyplot as plt

# plt.plot(np.cumsum(rets1) / (np.arange(len(rets1)) + 1), label="Q-Learning")
# plt.plot(np.cumsum(rets2) / (np.arange(len(rets2)) + 1), label="SARSA")
# plt.ylabel("Mean Return")
# plt.xlabel("Episode")
# plt.legend()
# plt.title("Q-Learning vs SARSA (Gridworld)")
# plt.grid()
# plt.show()
# from matplotlib import pyplot as plt

plt.imshow(Q1, cmap='cool', interpolation='nearest')
plt.colorbar(label="Q-value")
plt.title("Q-table for Q-learning (States x Actions)")
plt.xlabel("Action (0=Left, 1=Right)")
plt.ylabel("State")
plt.show()
