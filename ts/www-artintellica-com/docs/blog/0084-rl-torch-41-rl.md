+++
title = "Learn Reinforcement Learning with PyTorch, Part 4.1: What is Reinforcement Learning? RL vs. Supervised/Unsupervised"
author = "Artintellica"
date = "2025-06-13"
+++

## Introduction

At this point, you’ve mastered PyTorch and deep neural networks for supervised
tasks. But what happens when you don’t have a dataset of “inputs and
targets”—when learning is driven by **interaction** and **feedback**, not
labeled answers? Welcome to **Reinforcement Learning (RL)**.

In this article, you’ll:

- Understand what RL is and how it differs fundamentally from supervised and
  unsupervised learning.
- Simulate a simple agent/environment loop.
- Assign rewards to actions in a toy world.
- See and compare code patterns between supervised learning, unsupervised
  learning, and RL.
- Visualize how an agent’s experience and reward evolve over time.

---

## Mathematics: RL vs. Supervised/Unsupervised

**Supervised Learning:**

- You’re given a dataset $\{(x_i, y_i)\}_{i=1}^N$ of (input, target) pairs.
- The learning objective is usually to minimize:
  $$
  L_\text{sup} = \frac{1}{N} \sum_{i=1}^N \ell(f(x_i), y_i)
  $$
  where $\ell$ is a loss function (e.g., cross-entropy, MSE).

**Unsupervised Learning:**

- You’re only given $\{x_i\}_{i=1}^N$, no targets.
- You might organize, cluster, or compress data by optimizing:
  $$
  L_\text{unsup} = \frac{1}{N} \sum_{i=1}^N \ell_\text{unsup}(f(x_i))
  $$
  Examples: K-means, PCA, autoencoders.

**Reinforcement Learning:**

- An **agent** interacts with an **environment**, receiving state $s_t$, taking
  action $a_t$, getting reward $r_t$ and new state $s_{t+1}$:
  $$
  \text{Loop:} \quad s_t \xrightarrow{a_t} (r_t,\, s_{t+1})
  $$
- The agent’s objective is to _maximize total return_ (often, discounted sum of
  rewards):
  $$
  G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dotsb
  $$
- There is no ground truth "target" for each state—the agent _discovers_
  effective behaviors through feedback.

---

## Explanation: The Essence of RL

- **Supervised:** Learn to map $x$ to $y$, with direct ground truth, and
  “one-shot” feedback per data sample.
- **Unsupervised:** Find structure in $x$ alone (no $y$).
- **RL:** Learn by **trial and error** in a live loop; feedback (reward) is
  delayed and sparse, and the “right” actions may not be immediately obvious!

**Key RL workflow:**

- Start with an initial state $s_0$.
- Repeatedly, the agent chooses action $a_t$, gets reward $r_t$, moves to
  $s_{t+1}$ (from the environment).
- The agent “learns from experience,” adapting its policy to maximize _future_
  reward.

This is a natural fit for games, robotics, and many self-improving systems.

---

## Python Demonstrations

### Demo 1: Simulate an Agent-Environment Loop and Print State Transitions

Let’s make a tiny gridworld with 3 states (`0`, `1`, `2`), and two actions:
“right” and “left.” The agent can move right (`+1`) or left (`-1`). If it
reaches the end (`2`), it gets a reward of +1; at `0`, a penalty of -1.

```python
import random

class TinyEnv:
    def __init__(self) -> None:
        self.state: int = 1   # start in middle
        self.terminal: bool = False
    def reset(self) -> int:
        self.state = 1
        self.terminal = False
        return self.state
    def step(self, action: int) -> tuple[int, float, bool]:
        # action: 0=left, 1=right
        if action == 0:
            self.state -= 1
        else:
            self.state += 1
        if self.state <= 0:
            reward = -1.0
            self.terminal = True
            self.state = 0
        elif self.state >= 2:
            reward = +1.0
            self.terminal = True
            self.state = 2
        else:
            reward = 0.0
        return self.state, reward, self.terminal

env = TinyEnv()
state = env.reset()
print("start state:", state)
for t in range(10):
    action = random.choice([0, 1])        # random left/right
    new_state, reward, done = env.step(action)
    print(f"t={t}: action={action}, state={new_state}, reward={reward}")
    if done:
        print("Episode done!")
        break
```

---

### Demo 2: Assign Rewards to Agent Actions in a Toy Environment

Let’s try a hand-made policy: always go right!

```python
env = TinyEnv()
state = env.reset()
total_reward = 0
print("Policy: always go right")
for t in range(10):
    action = 1      # always right
    new_state, reward, done = env.step(action)
    print(f"t={t}: action={action}, state={new_state}, reward={reward}")
    total_reward += reward
    if done:
        print("Episode done with total reward:", total_reward)
        break
```

Try setting `action = 0` ("left") for a different outcome and see the reward!

---

### Demo 3: Compare Supervised, Unsupervised, and RL Code Snippets

**Supervised:**

```python
# Given (x, y) pairs, optimize for y
x_data = [0, 1, 2, 3]
y_data = [0, 1, 4, 9]
def supervised_train(x, y):
    model_output = [i**2 for i in x]
    loss = sum((y1 - y2)**2 for y1, y2 in zip(model_output, y))
    print("Supervised Loss:", loss)
supervised_train(x_data, y_data)
```

**Unsupervised:**

```python
# Given only x, cluster/group/transform
x_data = [0.5, 1.5, 3.2, 8.1, 9.3]
def unsupervised_task(x):
    center = sum(x)/len(x)
    clusters = [0 if xi < center else 1 for xi in x]
    print("Unsupervised clusters:", clusters)
unsupervised_task(x_data)
```

**Reinforcement Learning:**

```python
# Agent interacts and gets feedback as reward
env = TinyEnv()
state = env.reset()
total_reward = 0
for t in range(5):
    action = random.choice([0, 1])
    state, reward, done = env.step(action)
    total_reward += reward
    if done:
        break
print("RL: total reward from one trajectory:", total_reward)
```

---

### Demo 4: Plot Agent Total Reward Over Time for a Simple Policy

Let’s run many episodes and plot the cumulative reward a random agent gets:

```python
import matplotlib.pyplot as plt

episodes = 100
ep_rewards = []
for _ in range(episodes):
    env = TinyEnv()
    state = env.reset()
    total = 0
    for t in range(10):
        action = random.choice([0, 1])
        state, reward, done = env.step(action)
        total += reward
        if done:
            break
    ep_rewards.append(total)
plt.plot(ep_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Random Agent Reward over Episodes")
plt.grid(True)
plt.show()
```

Try swapping in a fixed policy (`action = 1`) to see how performance changes.

---

## Exercises

### **Exercise 1:** Simulate an Agent-Environment Loop and Print State Transitions

- Define a toy environment: finite states, a reset function, a step function
  that changes the state, and a termination condition.
- Have an agent take random actions, and print state, action, reward, and done
  status at every step.

---

### **Exercise 2:** Assign Rewards to Agent Actions in a Toy Environment

- Hardcode a simple policy (always left, always right, or alternate).
- Try different starting states or rules, and see how total reward or end state
  changes.

---

### **Exercise 3:** Compare Supervised, Unsupervised, and RL Code Snippets

- Write minimal code implementing a:
  - Supervised mapping ($x \rightarrow y$)
  - Unsupervised grouping (cluster)
  - RL feedback loop (action, reward, state transition)
- Discuss/observe the difference in inputs, outputs, and feedback.

---

### **Exercise 4:** Plot Agent's Total Reward Over Time for a Simple Policy

- For 50+ episodes, have the agent follow a fixed or random policy.
- Record total per-episode reward.
- Plot reward vs. episode, and see if any patterns/trends emerge.

---

### **Sample Starter Code for Exercises**

```python
import random
import matplotlib.pyplot as plt

class MiniEnv:
    def __init__(self) -> None:
        self.state = 1
        self.done = False
    def reset(self) -> int:
        self.state = 1
        self.done = False
        return self.state
    def step(self, action: int) -> tuple[int, float, bool]:
        if action == 0: self.state -= 1
        else: self.state += 1
        if self.state <= 0:
            self.state = 0; self.done = True; return self.state, -1.0, True
        if self.state >= 2:
            self.state = 2; self.done = True; return self.state, 1.0, True
        return self.state, 0.0, False

# Exercise 1/2
env = MiniEnv()
state = env.reset()
for t in range(8):
    action = 1 if t%2==0 else 0 # alternate
    next_state, reward, done = env.step(action)
    print(f"t={t}, state={next_state}, reward={reward}, done={done}")
    if done: break

# Exercise 3 - All three paradigms
# (see above Demos for code)

# Exercise 4
ep_rewards = []
for ep in range(60):
    env = MiniEnv()
    state = env.reset()
    total = 0
    for t in range(10):
        action = random.choice([0, 1])
        state, reward, done = env.step(action)
        total += reward
        if done: break
    ep_rewards.append(total)
plt.plot(ep_rewards)
plt.xlabel("Episode"); plt.ylabel("Total Reward")
plt.title("Random Policy, Per-Episode Reward"); plt.show()
```

---

## Conclusion

You now grasp the fundamental difference between RL, supervised, and
unsupervised learning—in workflow and in code. RL is about **trial, error,
reward, and improvement over time**, with no ground-truth answers. Everything
you’ve learned so far (from differentiation to neural nets) will help your agent
function, but _now it’s about learning from the loop!_ Next up: formalizing the
agent’s world with **Markov Decision Processes** and exploring core classic RL
methods.

See you in Part 4.2!
