import numpy as np
import random
import matplotlib.pyplot as plt

# Toy MDP setup
states = ["A", "B", "C"]
actions = ["left", "right"]
P = {
    "A": {"left": [(1.0, "A")], "right": [(1.0, "B")]},
    "B": {"left": [(0.8, "A"), (0.2, "C")], "right": [(1.0, "C")]},
    "C": {"left": [(1.0, "B")], "right": [(1.0, "C")]},
}
R = {
    "A": {"left": {"A": 0.0}, "right": {"B": 1.0}},
    "B": {"left": {"A": 0.0, "C": 2.0}, "right": {"C": 3.0}},
    "C": {"left": {"B": 0.0}, "right": {"C": 0.0}},
}


def sample_next_state(s, a):
    return random.choices([ns for _, ns in P[s][a]], [p for p, _ in P[s][a]])[0]


def get_reward(s, a, s2):
    return R[s][a][s2]


V_mc = {s: 0.0 for s in states}
V_td = {s: 0.0 for s in states}
gamma, alpha = 0.95, 0.12


def run_episode_MC(V):
    traj = []
    s = "A"
    for _ in range(7):
        a = random.choice(actions)
        s2 = sample_next_state(s, a)
        r = get_reward(s, a, s2)
        traj.append((s, r))
        s = s2
    G = 0
    for t in reversed(range(len(traj))):
        s, r = traj[t]
        G = r + gamma * G
        V[s] += alpha * (G - V[s])


def run_episode_TD(V):
    s = "A"
    for _ in range(7):
        a = random.choice(actions)
        s2 = sample_next_state(s, a)
        r = get_reward(s, a, s2)
        V[s] += alpha * (r + gamma * V[s2] - V[s])
        s = s2


# Track and plot (see above Demos)
