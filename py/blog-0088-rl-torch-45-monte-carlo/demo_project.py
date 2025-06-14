import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=True)
n_states = env.observation_space.n  # type: ignore
V_td = np.zeros(n_states)
gamma = 0.99
alpha = 0.08
episodes = 5000
history = []

for ep in range(episodes):
    s, _ = env.reset()
    done = False
    while not done:
        a = env.action_space.sample()  # random policy
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        V_td[s] += alpha * (float(r) + gamma * float(V_td[s2]) - V_td[s])
        s = s2
    history.append(V_td.copy())

history = np.array(history)
# Plot value trace for first 10 states
plt.figure(figsize=(8, 5))
for i in range(min(10, n_states)):
    plt.plot(history[:, i], label=f"State {i}")
plt.xlabel("Episode")
plt.ylabel("Value Estimate")
plt.legend()
plt.title("TD(0) Value Learning on FrozenLake")
plt.grid()
plt.show()

# Visualize value as an 4x4 grid (for small FrozenLake)
plt.figure(figsize=(3, 3))
plt.imshow(V_td.reshape(4, 4), cmap="viridis")
plt.colorbar(label="V(s)")
plt.title("Value Function (TD) for FrozenLake 4x4")
plt.show()
