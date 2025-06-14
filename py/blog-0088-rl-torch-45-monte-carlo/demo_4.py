import numpy as np
import random

states = ['A', 'B', 'C']
actions = ['left', 'right']
P = {
    'A': {'left': [(1.0, 'A')], 'right': [(1.0, 'B')]},
    'B': {'left': [(0.8, 'A'), (0.2, 'C')], 'right': [(1.0, 'C')]},
    'C': {'left': [(1.0, 'B')], 'right': [(1.0, 'C')]}
}
R = {
    'A': {'left': {'A': 0.0}, 'right': {'B': 1.0}},
    'B': {'left': {'A': 0.0, 'C': 2.0}, 'right': {'C': 3.0}},
    'C': {'left': {'B': 0.0}, 'right': {'C': 0.0}}
}
def sample_next_state(s,a):
    return random.choices([ns for _,ns in P[s][a]], [p for p,_ in P[s][a]])[0]
def get_reward(s,a,s2): return R[s][a][s2]
gamma = 0.95

# Initialize value tables
V_mc = {s: 0.0 for s in states}
V_td = {s: 0.0 for s in states}
alpha = 0.1

def run_episode_MC(V, epsilon=0.3):
    # Generate an episode using epsilon-random actions
    traj = []
    s = 'A'
    for t in range(6):
        a = random.choice(actions) if random.random() < epsilon else 'right'
        s2 = sample_next_state(s, a)
        r = get_reward(s, a, s2)
        traj.append((s, a, r))
        s = s2
    # Compute MC returns and update V
    G = 0.0
    for t in reversed(range(len(traj))):
        s, a, r = traj[t]
        G = r + gamma * G
        V[s] += alpha * (G - V[s])
    return list(V.values())

def run_episode_TD(V, epsilon=0.3):
    s = 'A'
    for t in range(6):
        a = random.choice(actions) if random.random() < epsilon else 'right'
        s2 = sample_next_state(s, a)
        r = get_reward(s,a,s2)
        V[s] += alpha * (r + gamma * V[s2] - V[s])
        s = s2
    return list(V.values())

mc_trace = []
td_trace = []
for ep in range(120):
    mc_trace.append(run_episode_MC(V_mc))
    td_trace.append(run_episode_TD(V_td))
mc_trace = np.array(mc_trace)
td_trace = np.array(td_trace)

import matplotlib.pyplot as plt
for i, s in enumerate(states):
    plt.plot(mc_trace[:,i], label=f"MC {s}")
    plt.plot(td_trace[:,i], '--', label=f"TD {s}")
plt.legend(); plt.xlabel("Episode")
plt.ylabel("Value Estimate")
plt.title("State Value Estimates: MC vs TD(0)")
plt.grid(); plt.show()

def greedy_policy(V):
    # Always pick the action leading to the next state with highest value
    policy = {}
    for s in states:
        best_a = max(actions, key=lambda a: sum(p * V[s2] for p, s2 in P[s][a]))
        policy[s] = best_a
    return policy

best_policy_mc = greedy_policy(V_mc)
best_policy_td = greedy_policy(V_td)
print("Greedy policy from MC values:", best_policy_mc)
print("Greedy policy from TD values:", best_policy_td)

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
        V_td[s] += alpha * (float( r ) + gamma * float( V_td[s2] ) - V_td[s])
        s = s2
    history.append(V_td.copy())

history = np.array(history)
# Plot value trace for first 10 states
plt.figure(figsize=(8,5))
for i in range(min(10, n_states)):
    plt.plot(history[:,i], label=f"State {i}")
plt.xlabel("Episode"); plt.ylabel("Value Estimate")
plt.legend(); plt.title("TD(0) Value Learning on FrozenLake")
plt.grid(); plt.show()

# Visualize value as an 4x4 grid (for small FrozenLake)
plt.figure(figsize=(3,3))
plt.imshow(V_td.reshape(4,4), cmap='viridis')
plt.colorbar(label="V(s)")
plt.title("Value Function (TD) for FrozenLake 4x4")
plt.show()
