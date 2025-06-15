+++
title = "Learn Reinforcement Learning with PyTorch, Part 4.7: Mini-Project—RL Agent on Custom Gridworld"
author = "Artintellica"
date = "2025-06-14"
code = "https://github.com/artintellica/artintellica/tree/main/py/blog-0090-rl-torch-47-project"
+++

Welcome to a hands-on mini-project at the heart of classic reinforcement
learning! In this post, we’ll tie together everything you’ve learned in Module 4
and build a simple, custom Gridworld environment from scratch. We’ll program an
RL agent to solve it with tabular Q-learning, visualize how the policy evolves,
and experiment with reward shaping to see how incentives impact behavior.

You’ll leave this post with a working project and an intuitive feel for how RL
agents learn to navigate environments of your own design.

---

## Contents

1. Intro: The Power of Custom Gridworlds
2. Mathematical Concepts: MDPs, the Gridworld, and Q-learning
3. From Math to Code: What Maps to What?
4. Python Demos:
   - Building a Gridworld Environment
   - Q-learning Implementation
   - Policy Visualization
5. Exercises:
   - Exercise 1: Custom Gridworld Environment
   - Exercise 2: Tabular Q-learning
   - Exercise 3: Policy Visualization
   - Exercise 4: Reward Shaping and Agent Behavior

---

## 1. Introduction: The Power of Custom Gridworlds

Classic gridworlds are RL’s “hello world”—simple, visual, and endlessly
tweakable. Designing your own environment forces you to deeply understand
states, actions, rewards, and how the Q-learning algorithm steers an agent to
success. You get direct feedback by watching your agent improve over time and
seeing your ideas about incentives play out.

By the end of this post, you’ll have a custom RL environment _and_ a tabular
Q-learning agent that you can modify and extend. This is where RL truly comes
alive!

---

## 2. Mathematical Concepts: MDPs, Gridworlds, and Q-learning

### Markov Decision Processes (MDPs)

A **Markov Decision Process (MDP)** is defined by the tuple
$(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ where:

- $\mathcal{S}$: set of states (e.g., all grid cells in our gridworld)
- $\mathcal{A}$: set of actions (e.g., UP, DOWN, LEFT, RIGHT)
- $P(s' \mid s, a)$: transition probability from $s$ to $s'$ when taking action
  $a$
- $R(s, a, s')$: reward for $(s, a, s')$
- $\gamma$: discount factor, $0 \le \gamma \le 1$

### The Gridworld

In a gridworld, the agent moves in a discrete rectangular grid. At each
timestep, it selects an action and transitions to a new cell (unless it hits a
wall or boundary). Some cells might be “goal” (high reward), some “pit”
(negative reward or episode ends), and the rest neutral (zero reward).

### Q-Learning: Tabular Value Update

**Q-learning** seeks to learn an optimal policy via the action-value function:

$$
Q(s, a) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t, s_{t+1}) \mid s_0 = s, a_0 = a, \pi^* \right]
$$

The Q-learning update rule for an observed transition $(s, a, r, s')$ is:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \big( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \big)
$$

where:

- $\alpha$ is the learning rate $(0 < \alpha \leq 1)$
- $r$ is the observed reward
- $\gamma$ is the discount factor

---

## 3. From Math to Code: What Maps to What?

- Each **grid cell** is a _state_ $s \in \mathcal{S}$
- Each direction (UP, DOWN, LEFT, RIGHT) is an _action_ $a \in \mathcal{A}$
- **Transition** is deterministic: action moves the agent, unless blocked
- **Reward table**: you define which cells are rewarding or penalizing
- **Q-table**: A 2D tensor or array $Q[s, a]$ mapping (state, action) pairs to
  Q-values

We’ll implement:

- A `GridworldEnv` class for the environment
- A Q-learning training loop
- Visualization tools to display learned policy and agent behavior

---

## 4. Python Demos

Let’s see the core code you’ll work with.

### Demo: Building a Simple Gridworld Environment

```python
import numpy as np
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt

class GridworldEnv:
    def __init__(
        self,
        width: int,
        height: int,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        pits: Optional[List[Tuple[int, int]]] = None,
        rewards: Optional[Dict[Tuple[int, int], float]] = None,
        walls: Optional[List[Tuple[int, int]]] = None,
        step_reward: float = -0.01,
    ) -> None:
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.pits = pits or []
        self.walls = walls or []
        self.rewards = rewards or {}
        self.step_reward = step_reward
        self.agent_pos = start
        # Action encoding: UP, RIGHT, DOWN, LEFT
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        self.n_actions = 4
        self.state_space = [(x, y)
                            for x in range(width)
                            for y in range(height)
                            if (x, y) not in self.walls]

    def reset(self) -> Tuple[int, int]:
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        dx, dy = self.actions[action]
        x, y = self.agent_pos
        new_x = min(max(x + dx, 0), self.width - 1)
        new_y = min(max(y + dy, 0), self.height - 1)
        next_pos = (new_x, new_y)

        # Blocked by wall
        if next_pos in self.walls:
            next_pos = self.agent_pos

        self.agent_pos = next_pos

        # Check terminal state
        done = False
        reward = self.rewards.get(next_pos, self.step_reward)

        if next_pos == self.goal:
            reward = 1.0
            done = True
        elif next_pos in self.pits:
            reward = -1.0
            done = True

        return next_pos, reward, done

    def render(self, policy: Optional[np.ndarray] = None) -> None:
        grid = np.full((self.height, self.width), ' ')
        for (x, y) in self.walls:
            grid[y, x] = '#'
        for (x, y) in self.pits:
            grid[y, x] = 'P'
        gx, gy = self.goal
        grid[gy, gx] = 'G'
        sx, sy = self.start
        grid[sy, sx] = 'S'
        ax, ay = self.agent_pos
        grid[ay, ax] = 'A'
        for y in range(self.height):
            line = ''
            for x in range(self.width):
                c = grid[y, x]
                if policy is not None and c == ' ':
                    s_idx = self.state_space.index((x, y))
                    a = np.argmax(policy[s_idx])
                    a_char = '↑→↓←'[a]
                    line += a_char
                else:
                    line += c
            print(line)
        print('')

    def state_to_idx(self, state: Tuple[int, int]) -> int:
        return self.state_space.index(state)

    def idx_to_state(self, idx: int) -> Tuple[int, int]:
        return self.state_space[idx]
```

---

### Demo: Tabular Q-learning Agent (PyTorch Not Needed for Table, but Used for Exercise Extension)

```python
import random

def q_learning(
    env: GridworldEnv,
    episodes: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
) -> Tuple[np.ndarray, List[float]]:
    n_states = len(env.state_space)
    n_actions = env.n_actions
    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    rewards_per_episode: List[float] = []

    for ep in range(episodes):
        state = env.reset()
        state_idx = env.state_to_idx(state)
        done = False
        total_reward = 0.0

        while not done:
            # Epsilon-greedy policy
            if random.random() < epsilon:
                action = random.randint(0, n_actions - 1)
            else:
                action = int(np.argmax(Q[state_idx]))

            next_state, reward, done = env.step(action)
            next_idx = env.state_to_idx(next_state)
            best_next_q = np.max(Q[next_idx])

            old_value = Q[state_idx, action]
            Q[state_idx, action] += alpha * (
                reward + gamma * best_next_q - Q[state_idx, action]
            )
            state_idx = next_idx
            total_reward += reward

        rewards_per_episode.append(total_reward)
    return Q, rewards_per_episode
```

---

## 5. Python Exercises

### **Exercise 1: Design a Custom Gridworld Environment in Python**

**Description:**  
Create a 5x5 gridworld with:

- Start state: (0, 0)
- Goal state: (4, 4)
- One pit at (2, 2)
- Walls at (1, 2) and (3, 1)
- Step reward: -0.05 per move

Add the cell (3, 3) as an extra reward cell that gives +0.5 if landed on.

Visualize the initial environment using the `render()` method.

```python
from typing import Tuple, List, Dict

def build_custom_gridworld() -> GridworldEnv:
    width, height = 5, 5
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (4, 4)
    pits: List[Tuple[int, int]] = [(2, 2)]
    walls: List[Tuple[int, int]] = [(1, 2), (3, 1)]
    rewards: Dict[Tuple[int, int], float] = {(3, 3): 0.5}
    step_reward: float = -0.05

    env = GridworldEnv(
        width=width,
        height=height,
        start=start,
        goal=goal,
        pits=pits,
        walls=walls,
        rewards=rewards,
        step_reward=step_reward,
    )
    return env

if __name__ == "__main__":
    env = build_custom_gridworld()
    env.render()
```

Sample output:

```
S   #
  #
  P
   G
```

---

### **Exercise 2: Implement Tabular Q-learning for Your Environment**

**Description:**  
Use the `q_learning()` function above. Set:

- $\alpha = 0.2$
- $\gamma = 0.95$
- $\epsilon = 0.2$
- $500$ episodes

After training, print the Q-table for inspection.

```python
import numpy as np

def print_q_table(env: GridworldEnv, Q: np.ndarray) -> None:
    for idx, state in enumerate(env.state_space):
        q_vals = Q[idx]
        print(f"State {state}: Q = {q_vals}")

if __name__ == "__main__":
    env = build_custom_gridworld()
    Q, rewards_per_episode = q_learning(
        env,
        episodes=500,
        alpha=0.2,
        gamma=0.95,
        epsilon=0.2,
    )
    print_q_table(env, Q)
```

---

### **Exercise 3: Train Agent and Visualize Policy After Each Episode**

**Description:**  
Visualize the greedy policy (argmax over Q) on the grid after every 100
episodes. Use arrows to display the agent's action preference at each cell.
(Optional: plot episode reward curve.)

```python
import matplotlib.pyplot as plt

def plot_rewards(rewards: List[float]) -> None:
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Reward Over Time")
    plt.show()

if __name__ == "__main__":
    env = build_custom_gridworld()
    Q = np.zeros((len(env.state_space), env.n_actions), dtype=np.float32)
    all_rewards: List[float] = []
    episodes = 500
    intervals = [100, 200, 300, 400, 500]

    for episode in range(1, episodes + 1):
        state = env.reset()
        state_idx = env.state_to_idx(state)
        done = False
        total_reward = 0.0

        while not done:
            # Epsilon-greedy
            if random.random() < 0.2:
                action = random.randint(0, env.n_actions - 1)
            else:
                action = int(np.argmax(Q[state_idx]))

            next_state, reward, done = env.step(action)
            next_idx = env.state_to_idx(next_state)
            best_next_q = np.max(Q[next_idx])
            # Q-learning update
            Q[state_idx, action] += 0.2 * (reward + 0.95 * best_next_q - Q[state_idx, action])
            state_idx = next_idx
            total_reward += reward

        all_rewards.append(total_reward)

        if episode in intervals:
            print(f"Policy after episode {episode}:")
            policy = Q.copy()
            env.render(policy=policy)

    plot_rewards(all_rewards)
```

---

### **Exercise 4: Modify Reward Structure and Observe Learned Policy Changes**

**Description:**  
Change the reward for (3, 3) to -0.5 (now it’s a “trap” cell). Retrain the agent
from scratch and compare the learned policy with the original. Did the agent
learn to avoid (3, 3)?

**Code:**

```python
def build_gridworld_with_trap() -> GridworldEnv:
    width, height = 5, 5
    start = (0, 0)
    goal = (4, 4)
    pits = [(2, 2)]
    walls = [(1, 2), (3, 1)]
    rewards = {(3, 3): -0.5}  # Now a trap!
    step_reward = -0.05

    env = GridworldEnv(
        width=width,
        height=height,
        start=start,
        goal=goal,
        pits=pits,
        walls=walls,
        rewards=rewards,
        step_reward=step_reward,
    )
    return env

if __name__ == "__main__":
    print("Training on new environment with trap...")
    env_trap = build_gridworld_with_trap()
    Q_trap, rewards_trap = q_learning(
        env_trap,
        episodes=500,
        alpha=0.2,
        gamma=0.95,
        epsilon=0.2,
    )
    print("Learned policy (with trap):")
    env_trap.render(policy=Q_trap)
    plot_rewards(rewards_trap)
```

---

## Conclusion

Congratulations! By designing your own Gridworld, training a tabular Q-learning
agent, and visualizing both the policy and how incentives affect learning,
you’ve gained a “systems-level” understanding of classical RL. These building
blocks are the kernel of every RL problem, from simple games to AI for robotics.

Experiment more: change grid size, reward layout, wall configurations, or even
make stochastic transitions. Next, we’ll bring all this together with function
approximation and deep networks—stay tuned!
