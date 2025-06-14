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
