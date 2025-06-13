from typing import Dict, Tuple, List
import numpy as np

states: List[str] = ["A", "B", "C"]
actions: List[str] = ["left", "right"]

# Transition probabilities: P[s][a] = list of (prob, next_state)
P: Dict[str, Dict[str, List[Tuple[float, str]]]] = {
    "A": {"left": [(1.0, "A")], "right": [(1.0, "B")]},
    "B": {"left": [(0.8, "A"), (0.2, "C")], "right": [(1.0, "C")]},
    "C": {"left": [(1.0, "B")], "right": [(1.0, "C")]},
}

# Rewards: R[s][a][s'] = reward
R: Dict[str, Dict[str, Dict[str, float]]] = {
    "A": {"left": {"A": 0.0}, "right": {"B": 1.0}},
    "B": {"left": {"A": 0.0, "C": 2.0}, "right": {"C": 3.0}},
    "C": {"left": {"B": 0.0}, "right": {"C": 0.0}},
}
gamma: float = 0.9

import random


def sample_next_state(s: str, a: str) -> str:
    p_list = P[s][a]
    probs, next_states = zip(*[(p, s2) for p, s2 in p_list])
    return random.choices(next_states, weights=probs)[0]


def get_reward(s: str, a: str, s2: str) -> float:
    return R[s][a][s2]


def run_episode(policy: Dict[str, str] | None = None, max_steps: int = 10) -> float:
    s = "A"
    total_reward = 0.0
    trajectory = []
    for t in range(max_steps):
        a = policy[s] if policy and policy[s] else random.choice(actions)
        s2 = sample_next_state(s, a)
        r = get_reward(s, a, s2)
        total_reward += r
        trajectory.append((s, a, r, s2))
        s = s2
    print("Trajectory:", trajectory)
    print("Total reward:", total_reward)
    return total_reward


ep_rewards = [run_episode() for _ in range(5)]
print("Average reward:", np.mean(ep_rewards))

import networkx as nx
import matplotlib.pyplot as plt

G = nx.MultiDiGraph()
for s in P:
    for a in P[s]:
        for prob, s2 in P[s][a]:
            label = f"{a} ({prob})"
            G.add_edge(s, s2, label=label)

pos = nx.spring_layout(G, seed=1)
plt.figure(figsize=(5, 4))
nx.draw(
    G, pos, with_labels=True, node_size=1200, node_color="skyblue", font_weight="bold"
)
edge_labels = {(u, v, k["label"]): k["label"] for u, v, k in G.edges(data=True)}
edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
plt.title("MDP State Transition Graph")
plt.show()
