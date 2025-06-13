import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3", render_mode="ansi")
n_states = env.observation_space.n
n_actions = env.action_space.n

Q = np.zeros((n_states, n_actions))
episodes = 1800
alpha = 0.1
gamma = 0.98
eps = 0.15

lengths = []
for ep in range(episodes):
    s, _ = env.reset()
    done = False
    count = 0
    while not done:
        if np.random.rand() < eps:
            a = np.random.randint(n_actions)
        else:
            a = np.argmax(Q[s])
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])
        s = s2
        count += 1
    lengths.append(count)

import matplotlib.pyplot as plt
plt.plot(np.convolve(lengths, np.ones(50)/50, mode='valid'))
plt.xlabel("Episode")
plt.ylabel("Episode Length")
plt.title("Taxi-v3: Q-Learning Episode Length (lower is better)")
plt.show()
print("Mean episode length (last 100 episodes):", np.mean(lengths[-100:]))
