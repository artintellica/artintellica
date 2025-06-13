import random
from typing import Dict, Tuple, List
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Exercise 1
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


# Exercise 2
def run_episode(policy=None, max_steps=8):
    s = "A"
    total = 0
    for t in range(max_steps):
        a = policy[s] if policy else random.choice(actions)
        s2 = sample_next_state(s, a)
        r = get_reward(s, a, s2)
        print(f"t={t}, s={s}, a={a}, r={r}, s'={s2}")
        total += r
        s = s2
    print("Total reward:", total)
    return total


run_episode()

# Exercise 3
G = nx.MultiDiGraph()
for s in P:
    for a in P[s]:
        for prob, s2 in P[s][a]:
            lbl = f"{a} ({prob})"
            G.add_edge(s, s2, label=lbl)
pos = nx.spring_layout(G, seed=2)
plt.figure(figsize=(5, 4))
nx.draw(G, pos, with_labels=True, node_size=1300, node_color="orange")
edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("State Transitions")
plt.show()

# Exercise 4
right_policy = {s: "right" for s in states}
left_policy = {s: "left" for s in states}
rewards_right = [run_episode(right_policy) for _ in range(10)]
rewards_left = [run_episode(left_policy) for _ in range(10)]
print("Avg reward right:", np.mean(rewards_right))
print("Avg reward left:", np.mean(rewards_left))
