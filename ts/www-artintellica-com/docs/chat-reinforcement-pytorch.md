+++
model = "o3"
+++

i need your help to produce a series of blog posts. we are learning machine learning, and we are doing it with pytorch. our focus is on reinforcement learning. the series is called "Learn Reinforcement Learning with PyTorch".

i need your help to write each blog post. there should always be an intro, an overview of mathematical concepts including definitions, a section connecting the math to the code, and then a series of python demos, and then a series of python exercises.

always include the code for the exercises. put the full code for the python exercises after the description of each exercise. no need to hide the code from the reader.

for all python code, always use fully typed code with type hints.

each blog post should always use dollar-denominated math equations. for instance, $V(s) = \mathbb{E}[G_t \mid s_t = s]$ for the state value function, where $G_t$ is the return starting from time $t$.

block-level math equations should be in double dollar signs, like this:

$$
V(s) = \mathbb{E}[G_t \mid s_t = s]
$$

where $G_t$ is the return starting from time $t$.

always put front-matter at the top o the blog post, including title, author, and date, like this:

+++
title = "Learn Reinforcement Learning with PyTorch, Part N.M: [Title of the Post]"
author = "Artintellica"
date = "2025-06-15" [or whatever today's date is]
+++

the outline of the full course is as follows:

# **Course Outline: Learning Reinforcement Learning with PyTorch**

---

## **Module 1: Foundations—Vectors, Matrices, and PyTorch Basics**

### 1. Introduction: The Course, the Blog, and Why PyTorch

* **Exercise 1:** Install PyTorch and verify the installation.
* **Exercise 2:** Create your first tensor and print its shape and dtype.
* **Exercise 3:** Move a tensor between CPU and (if available) GPU/MPS, and print device info.
* **Exercise 4:** Run and explain a “Hello World” example—elementwise addition in PyTorch.

### 2. Vectors and Scalars—Hands-On with PyTorch Tensors

* **Exercise 1:** Create scalar and 1D vector tensors; change their shape.
* **Exercise 2:** Index and slice vectors; reverse a vector.
* **Exercise 3:** Compute mean, sum, and standard deviation of a vector.
* **Exercise 4:** Plot a vector as a line graph using Matplotlib.

### 3. Basic Vector Operations: Addition, Scalar Multiplication, Dot Product

* **Exercise 1:** Implement vector addition using PyTorch.
* **Exercise 2:** Scale a vector by a scalar (both manual and using broadcasting).
* **Exercise 3:** Compute the dot product of two vectors (manual and built-in).
* **Exercise 4:** Visualize vector addition and scalar multiplication in 2D with Matplotlib.

### 4. Matrices: Construction, Shapes, and Basic Matrix Operations

* **Exercise 1:** Create 2D tensors (matrices) with specific shapes.
* **Exercise 2:** Transpose a matrix and verify with PyTorch.
* **Exercise 3:** Perform elementwise addition and multiplication on two matrices.
* **Exercise 4:** Demonstrate broadcasting with a matrix and a vector.

### 5. Broadcasting and Elementwise Operations in PyTorch

* **Exercise 1:** Add a row vector to each row of a matrix using broadcasting.
* **Exercise 2:** Multiply a matrix by a column vector using broadcasting.
* **Exercise 3:** Identify and fix a broadcasting error in code.
* **Exercise 4:** Compare manual elementwise operations with PyTorch’s built-in operators.

### 6. Matrix Multiplication and Transpose—What, Why, and How

* **Exercise 1:** Multiply two matrices using `@` and `torch.matmul`.
* **Exercise 2:** Implement matrix multiplication “by hand” using loops and compare with PyTorch.
* **Exercise 3:** Visualize the effect of transposing a matrix on its shape and data.
* **Exercise 4:** Explain and fix a common shape-mismatch error in matmul.

### 7. Geometry with Tensors: Norms, Distance, Angles, Projections

* **Exercise 1:** Compute the Euclidean norm (length) of a vector.
* **Exercise 2:** Find the distance between two vectors.
* **Exercise 3:** Calculate the cosine similarity between two vectors.
* **Exercise 4:** Project one vector onto another and plot both.

### 8. Linear Transformations and Simple Data Transformations

* **Exercise 1:** Create rotation and scaling matrices.
* **Exercise 2:** Apply a rotation matrix to a set of 2D points.
* **Exercise 3:** Visualize the effect of a transformation on a shape (e.g., a square).
* **Exercise 4:** Chain multiple transformations and describe the result.

### 9. Hands-On Mini-Project: Visualizing and Transforming Data with Tensors

* **Exercise 1:** Load a simple 2D dataset (or generate synthetic data).
* **Exercise 2:** Apply various matrix transformations and visualize before/after.
* **Exercise 3:** Compute the mean and covariance matrix of the dataset.
* **Exercise 4:** Center and normalize the data with PyTorch.

---

## **Module 2: Optimization and Learning—From Gradients to Linear Models**

### 1. Introduction to Gradient Descent—Math and Code

* **Exercise 1:** Implement scalar gradient descent for a simple function (e.g., \$f(x) = x^2\$).
* **Exercise 2:** Visualize the optimization path on a 2D plot.
* **Exercise 3:** Use vector gradient descent on \$f(\mathbf{x}) = ||\mathbf{x}||^2\$.
* **Exercise 4:** Experiment with different learning rates and observe convergence.

### 2. Autograd in PyTorch: Automatic Differentiation Demystified

* **Exercise 1:** Mark a tensor as `requires_grad=True` and compute gradients.
* **Exercise 2:** Calculate gradients for a multi-variable function (e.g., \$f(x, y) = x^2 + y^3\$).
* **Exercise 3:** Zero gradients and explain why this is necessary in training loops.
* **Exercise 4:** Manually verify PyTorch’s computed gradients for a small example.

### 3. Loss Functions and Cost Surfaces—Visual and Practical Intuition

* **Exercise 1:** Implement Mean Squared Error (MSE) loss manually and with PyTorch.
* **Exercise 2:** Plot the loss surface for a simple linear model.
* **Exercise 3:** Visualize binary cross-entropy loss as a function of its input.
* **Exercise 4:** Compare how different loss functions penalize outliers.

### 4. Fitting a Line: Linear Regression from Scratch with PyTorch

* **Exercise 1:** Generate synthetic linear data with noise.
* **Exercise 2:** Implement linear regression training loop from scratch (using only tensors).
* **Exercise 3:** Use PyTorch’s autograd and optimizer to fit a line.
* **Exercise 4:** Plot predictions vs. ground truth and compute R² score.

### 5. Fitting Nonlinear Curves: Polynomial Regression

* **Exercise 1:** Generate polynomial data with noise.
* **Exercise 2:** Fit a polynomial using explicit feature construction in PyTorch.
* **Exercise 3:** Compare performance of linear vs. polynomial regression.
* **Exercise 4:** Visualize overfitting by increasing the polynomial degree.

### 6. Classification Basics: Logistic Regression

* **Exercise 1:** Generate binary classification data.
* **Exercise 2:** Implement logistic regression “from scratch” (sigmoid and BCE loss).
* **Exercise 3:** Train with PyTorch’s optimizer and compare results.
* **Exercise 4:** Plot decision boundary and accuracy.

### 7. Softmax and Multiclass Classification

* **Exercise 1:** Generate synthetic data for three classes.
* **Exercise 2:** Implement softmax and cross-entropy loss manually.
* **Exercise 3:** Train a multiclass classifier with PyTorch’s `nn.CrossEntropyLoss`.
* **Exercise 4:** Plot the class boundaries in 2D.

### 8. Mini-Project: Build, Train, and Visualize a Simple Classifier

* **Exercise 1:** Choose or generate a 2D dataset for classification.
* **Exercise 2:** Train a classifier (logistic or neural network) on the data.
* **Exercise 3:** Visualize the learned decision boundary.
* **Exercise 4:** Evaluate model accuracy and plot misclassified points.

---

## **Module 3: Neural Networks—Building Blocks and Training**

### 1. The Perceptron: Oldest Neural Network

* **Exercise 1:** Implement the perceptron update rule for a binary task.
* **Exercise 2:** Train the perceptron on linearly separable data.
* **Exercise 3:** Visualize decision boundary after each epoch.
* **Exercise 4:** Test on non-separable data and analyze behavior.

### 2. Feedforward Neural Networks from Scratch (No nn.Module)

* **Exercise 1:** Implement a two-layer neural network using tensors and matrix ops.
* **Exercise 2:** Forward propagate and compute loss for a batch of inputs.
* **Exercise 3:** Backpropagate gradients manually (by hand or using PyTorch’s autograd).
* **Exercise 4:** Train the model and plot loss over epochs.

### 3. Building with `torch.nn`: The Convenient Way

* **Exercise 1:** Rewrite previous NN using `nn.Module`.
* **Exercise 2:** Compare number of lines and readability.
* **Exercise 3:** Add a hidden layer and train on data.
* **Exercise 4:** Save and load model weights.

### 4. Activation Functions: Sigmoid, Tanh, ReLU, LeakyReLU, etc.

* **Exercise 1:** Plot different activation functions.
* **Exercise 2:** Train small NNs with each activation on the same task; compare convergence.
* **Exercise 3:** Observe vanishing/exploding gradients by visualizing gradients.
* **Exercise 4:** Swap activation mid-training and observe changes.

### 5. Backpropagation: Intuition and Hands-On Example

* **Exercise 1:** Compute gradients for a two-layer network by hand for a single example.
* **Exercise 2:** Use `.backward()` to compare with manual gradients.
* **Exercise 3:** Visualize gradient flow in the network.
* **Exercise 4:** Debug and fix a model with vanishing gradients.

### 6. Overfitting, Underfitting, and Regularization

* **Exercise 1:** Fit a model to small and large datasets and plot train/test loss.
* **Exercise 2:** Add noise to data and visualize overfitting.
* **Exercise 3:** Apply L2 regularization and observe effect.
* **Exercise 4:** Vary model complexity and record accuracy.

### 7. Dropout, L2, and Other Regularization in PyTorch

* **Exercise 1:** Add dropout layers to a network and plot training/validation accuracy.
* **Exercise 2:** Use weight decay with an optimizer.
* **Exercise 3:** Compare models with and without regularization.
* **Exercise 4:** Experiment with different dropout rates and interpret results.

### 8. Mini-Project: MNIST Digit Classifier (Shallow NN)

* **Exercise 1:** Download and load the MNIST dataset using PyTorch.
* **Exercise 2:** Build and train a shallow NN on MNIST.
* **Exercise 3:** Plot confusion matrix of predictions.
* **Exercise 4:** Visualize a few correctly and incorrectly classified digits.

---

## **Module 4: Introduction to Reinforcement Learning—Concepts & Classic Problems**

### 1. What is Reinforcement Learning? RL vs. Supervised/Unsupervised

* **Exercise 1:** Simulate an agent-environment loop and print state transitions.
* **Exercise 2:** Assign rewards to agent actions in a toy environment.
* **Exercise 3:** Compare supervised, unsupervised, and RL code snippets.
* **Exercise 4:** Plot agent total reward over time for a simple policy.

### 2. Markov Decision Processes: States, Actions, Rewards, Policies

* **Exercise 1:** Define a small MDP as Python data structures.
* **Exercise 2:** Simulate a random policy in the MDP and track rewards.
* **Exercise 3:** Visualize state transitions as a graph.
* **Exercise 4:** Implement a simple policy and compute expected reward.

### 3. Bandit Problems: Exploration vs. Exploitation

* **Exercise 1:** Simulate a multi-armed bandit and plot arm payout distributions.
* **Exercise 2:** Implement epsilon-greedy policy and track arm selections.
* **Exercise 3:** Compare UCB and Thompson Sampling on the same problem.
* **Exercise 4:** Plot cumulative reward for different strategies.

  * *Project Exercise:* Visualize learning curve of a bandit agent over time.

### 4. Tabular Value-Based Methods: Q-Learning and SARSA

* **Exercise 1:** Implement a Q-table for a small discrete environment.
* **Exercise 2:** Run Q-learning and SARSA on Gridworld, compare convergence.
* **Exercise 3:** Visualize the learned Q-table as a heatmap.
* **Exercise 4:** Animate agent’s trajectory using the learned policy.

  * *Project Exercise:* Train a Q-learning agent on Taxi-v3 and report average episode length.

### 5. Monte Carlo and Temporal Difference (TD) Learning

* **Exercise 1:** Implement MC and TD(0) value updates for a toy MDP.
* **Exercise 2:** Track value estimates over episodes and plot.
* **Exercise 3:** Compare convergence of MC and TD(0).
* **Exercise 4:** Use value estimates to improve policy and evaluate performance.

  * *Project Exercise:* Visualize value function estimates on FrozenLake-v1.

### 6. Policies, Value Functions, and Bellman Equations

* **Exercise 1:** Solve a small Bellman equation system by hand and with PyTorch.
* **Exercise 2:** Compute state-value and action-value functions for a policy.
* **Exercise 3:** Evaluate a random and a greedy policy in an MDP.
* **Exercise 4:** Visualize policy and value function in a gridworld.

### 7. Mini-Project: RL Agent on Custom Gridworld

* **Exercise 1:** Design a custom Gridworld environment in Python.
* **Exercise 2:** Implement tabular Q-learning for your environment.
* **Exercise 3:** Train agent and visualize policy after each episode.
* **Exercise 4:** Modify reward structure and observe changes in learned policy.

---

## **Module 5: Deep Reinforcement Learning—From Q-Tables to Deep Networks**

### 1. Limitations of Tabular RL and the Need for Function Approximation

* **Exercise 1:** Create a large state space where Q-tables are impractical.
* **Exercise 2:** Visualize exponential state growth in a simple environment.
* **Exercise 3:** Attempt tabular learning and analyze why it fails.
* **Exercise 4:** Propose a neural network architecture to replace Q-table.

### 2. Deep Q-Networks (DQN): Concepts and PyTorch Implementation

* **Exercise 1:** Build a Q-network using `nn.Module` for CartPole-v1.
* **Exercise 2:** Implement an experience replay buffer from scratch.
* **Exercise 3:** Train DQN agent and plot episode rewards.
* **Exercise 4:** Save and load trained DQN models.

  * *Project Exercise:* Experiment with different network sizes and plot learning curves.

### 3. Experience Replay and Target Networks

* **Exercise 1:** Implement a target network for stable Q-value updates.
* **Exercise 2:** Compare learning with and without experience replay.
* **Exercise 3:** Tune replay buffer size and target update frequency.
* **Exercise 4:** Analyze training stability through reward variance plots.

### 4. Extensions: Double DQN, Dueling DQN

* **Exercise 1:** Extend your DQN to Double DQN and compare performance.
* **Exercise 2:** Implement dueling network architecture in PyTorch.
* **Exercise 3:** Train both models on LunarLander-v2 and compare results.
* **Exercise 4:** Visualize Q-value distributions during training.

  * *Project Exercise:* Plot side-by-side learning curves for DQN, Double DQN, and Dueling DQN.

### 5. Policy Gradient Methods: REINFORCE Algorithm

* **Exercise 1:** Implement a policy network for CartPole-v1.
* **Exercise 2:** Code the REINFORCE algorithm and train agent.
* **Exercise 3:** Visualize total reward per episode and smoothed average.
* **Exercise 4:** Experiment with reward discounting and report effects.

  * *Project Exercise:* Show learned policy in action using environment render.

### 6. Advantage Actor-Critic (A2C/A3C) - Concepts & Simple Implementation

* **Exercise 1:** Implement separate actor and critic networks in PyTorch.
* **Exercise 2:** Train A2C agent on CartPole-v1.
* **Exercise 3:** Plot both value loss and policy loss over training.
* **Exercise 4:** Visualize the advantage estimates during episodes.

---

**This outline provides exercises for every post—concrete, PyTorch-based, and always hands-on. Ready to flesh out an individual blog post, or need code for any exercise? Just ask!**

---

please produce the blog post for section 5.4, whose description is:

### 4. Extensions: Double DQN, Dueling DQN

* **Exercise 1:** Extend your DQN to Double DQN and compare performance.
* **Exercise 2:** Implement dueling network architecture in PyTorch.
* **Exercise 3:** Train both models on LunarLander-v2 and compare results.
* **Exercise 4:** Visualize Q-value distributions during training.

  * *Project Exercise:* Plot side-by-side learning curves for DQN, Double DQN, and Dueling DQN.

please respond with the blog post only.

# === ASSISTANT ===

+++
title = "Learn Reinforcement Learning with PyTorch, Part 5.4: Extensions—Double DQN and Dueling DQN"
author = "Artintellica"
date = "2024-06-16"
+++

# Learn Reinforcement Learning with PyTorch, Part 5.4: Extensions—Double DQN and Dueling DQN

## Introduction

Advancements in Deep Q-Learning have brought us remarkable performance in complex reinforcement learning (RL) environments. However, standard DQN (Deep Q-Network) displays some well-known limitations, including overestimating action values and inefficient representation of state-value and action-advantages. In this blog post, we introduce two essential extensions that address these problems—**Double DQN** and **Dueling DQN**. We'll cover the mathematical motivation, provide practical PyTorch code, and guide you through hands-on exercises to train, compare, and visualize these improved algorithms on the [LunarLander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) environment.

---

## Mathematical Overview

### Q-Learning Recap

Q-Learning seeks to estimate the optimal action-value function:

$$
Q^*(s, a) = \max_\pi \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a, \pi \right]
$$

where $r_t$ is the reward at time $t$, $\gamma$ is the discount factor, and $\pi$ is a policy.

**DQN** uses deep neural networks as a function approximator for $Q(s, a;\,\theta)$ and mini-batch Q-learning updates:

$$
y^\text{DQN} = r + \gamma \max_{a'} Q_\text{target}(s', a';\,\theta^-) 
$$

where $\theta^-$ are the parameters of the target network.

### Double DQN: Reducing Overestimation Bias

DQN tends to overestimate action values due to the **max operator** in the target. Double DQN fixes this using separate networks for selection and evaluation:

$$
y^\text{Double DQN} = r + \gamma Q_\text{target}\big(s', \underset{a'}{\arg\max}\ Q_\text{online}(s', a';\,\theta),\ \theta^- \big)
$$

- $\theta$: parameters of the online network
- $\theta^-$: parameters of the target network

Here, **action selection** uses the online network, and **action evaluation** uses the target.

### Dueling DQN: Separating State-Value and Advantage

Dueling DQN proposes a new architecture:

$$
Q(s, a;\,\theta, \alpha, \beta) = V(s;\,\theta, \beta) + \left( A(s, a;\,\theta, \alpha) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a';\,\theta, \alpha) \right)
$$

- $V(s)$: Value function of state
- $A(s, a)$: Advantage function for action $a$ in state $s$

This separation helps the network learn which states are valuable irrespective of the action taken, improving training speed and performance in some environments.

---

## Connecting the Math to the Code

- **Double DQN**:
  - When calculating Q-learning targets, select the greedy action using the **online** (current) network, but evaluate its Q-value using the **target** network.
- **Dueling DQN**:
  - Design your Q-network such that, after some shared layers, you create two streams:
    - One outputs the scalar state value $V(s)$.
    - The other outputs the advantages $A(s, a)$ for all actions.
  - These are recombined as per the formula above to produce the Q-values for each action.

We'll provide full PyTorch implementations as hands-on Python demos and exercises below.

---

## Python Demos

Let's start with the essential building blocks. These will be reused in the exercises that follow.

### Demo 1: Dueling DQN Network Class

```python
import torch
import torch.nn as nn
from typing import Tuple

class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Scalar V(s)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # A(s, a) for each action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals  # shape: (batch, action_dim)
```

---

## Exercises

### Exercise 1: Extend your DQN to Double DQN and Compare Performance

**Description:**
Modify your existing DQN code to implement Double DQN. Specifically, during the target calculation, use the online network to select the next action, but use the target network to evaluate its Q-value.

**Full Double DQN target computation with fully typed PyTorch code:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

def compute_double_dqn_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_states: torch.Tensor,
    gamma: float,
    online_net: nn.Module,
    target_net: nn.Module
) -> torch.Tensor:
    """
    Compute Double DQN targets.
    
    Args:
        rewards: (batch, )
        dones: (batch, )
        next_states: (batch, state_dim)
        gamma: Discount factor
        online_net: The current DQN/dueling-DQN network
        target_net: The target network
    
    Returns:
        Q_targets: shape (batch, )
    """
    with torch.no_grad():
        next_q_online = online_net(next_states)  # (batch, action_dim)
        next_actions = torch.argmax(next_q_online, dim=1)  # (batch, )
        next_q_target = target_net(next_states)  # (batch, action_dim)
        next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (batch, )
        # Q_target = r + gamma * Q(s', a*) * (1 - done)
        Q_targets = rewards + gamma * next_q_values * (1 - dones)
    return Q_targets
```

---

### Exercise 2: Implement Dueling Network Architecture in PyTorch

**Description:**
Write a complete Dueling DQN network class (if you haven't done so in the demo already!), and compare its outputs with a standard MLP DQN model on random input.

**Full code:**

```python
import torch
import torch.nn as nn
from typing import Tuple

class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class StandardDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Test on random input
if __name__ == "__main__":
    state_dim: int = 8   # for LunarLander-v2
    action_dim: int = 4
    x: torch.Tensor = torch.randn(5, state_dim)

    dueling_net = DuelingDQN(state_dim, action_dim)
    standard_net = StandardDQN(state_dim, action_dim)
    print("Dueling DQN output:", dueling_net(x))
    print("Standard DQN output:", standard_net(x))
```

---

### Exercise 3: Train Both Models on LunarLander-v2 and Compare Results

**Description:**
Train both DQN/Double DQN and Dueling DQN agents on [LunarLander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/). Track and plot average episode rewards for each.

**Full code for a minimal training loop (assumes experience replay implementation):**

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from typing import Any, Deque, List, Tuple

# Define replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque = deque(maxlen=capacity)

    def push(self, *args: Any) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        batch = Transition(*zip(*batch))
        states      = torch.FloatTensor(np.array(batch.state))
        actions     = torch.LongTensor(np.array(batch.action))
        rewards     = torch.FloatTensor(np.array(batch.reward))
        next_states = torch.FloatTensor(np.array(batch.next_state))
        dones       = torch.FloatTensor(np.array(batch.done))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

# DuelingDQN and StandardDQN from previous code

# Hyperparameters
ENV_NAME: str = "LunarLander-v2"
GAMMA: float = 0.99
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
TARGET_UPDATE: int = 1000
REPLAY_SIZE: int = 50000
MIN_REPLAY_SIZE: int = 1000
EPSILON_START: float = 1.0
EPSILON_END: float = 0.02
EPSILON_DECAY: int = 10000
EPISODES: int = 500

def select_action(state: np.ndarray, policy_net: nn.Module, epsilon: float, action_dim: int) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(action_dim)
    else:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_vals = policy_net(state_t)
        return int(torch.argmax(q_vals, dim=1).item())

def train_lander(model_type: str = "double", dueling: bool = False, plot: bool = True) -> List[float]:
    env = gym.make(ENV_NAME)
    state_dim: int = env.observation_space.shape[0]
    action_dim: int = env.action_space.n
    if dueling:
        policy_net = DuelingDQN(state_dim, action_dim)
        target_net = DuelingDQN(state_dim, action_dim)
    else:
        policy_net = StandardDQN(state_dim, action_dim)
        target_net = StandardDQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay = ReplayBuffer(REPLAY_SIZE)

    steps_done = 0
    epsilon = EPSILON_START
    rewards_per_episode: List[float] = []

    state, _ = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.push(state, action, reward, next_state, float(done))
        state = next_state if not done else env.reset()[0]

    for ep in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            # Epsilon-greedy action
            epsilon = np.interp(steps_done, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
            action = select_action(state, policy_net, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            # Training step
            if len(replay) > BATCH_SIZE:
                batch = replay.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = batch

                # DQN/Double DQN target calculation
                with torch.no_grad():
                    if model_type == "double":
                        # Double DQN update
                        next_q_online = policy_net(next_states)
                        next_actions = torch.argmax(next_q_online, dim=1)
                        next_q_target = target_net(next_states)
                        next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    else:
                        # Standard DQN update
                        next_q_target = target_net(next_states)
                        next_q_values, _ = next_q_target.max(dim=1)

                    targets = rewards + GAMMA * next_q_values * (1 - dones)

                q_values = policy_net(states)
                action_qs = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(action_qs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target net
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            steps_done += 1

        rewards_per_episode.append(total_reward)
        if (ep + 1) % 10 == 0:
            mean_reward = np.mean(rewards_per_episode[-10:])
            print(f"Episode {ep+1}, mean reward (last 10): {mean_reward:.2f}")

    env.close()
    if plot:
        plt.plot(rewards_per_episode)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"{'Dueling ' if dueling else ''}{model_type.title()} DQN on LunarLander-v2")
        plt.show()
    return rewards_per_episode

# Run & Plot
# Standard DQN
rewards_dqn = train_lander(model_type="dqn", dueling=False)
# Double DQN
rewards_double = train_lander(model_type="double", dueling=False)
# Dueling Double DQN
rewards_dueling = train_lander(model_type="double", dueling=True)

# Side-by-side plot
plt.plot(rewards_dqn, label="DQN")
plt.plot(rewards_double, label="Double DQN")
plt.plot(rewards_dueling, label="Dueling Double DQN")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.title("LunarLander-v2: DQN vs Double DQN vs Dueling DQN")
plt.show()
```

---

### Exercise 4: Visualize Q-Value Distributions During Training

**Description:**
During or after training, visualize the distribution (histogram) of Q-values predicted by your DQN/Double DQN/Dueling DQN for a set of states. Repeat periodically to observe stability and learning characteristics.

**Full code:**

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_q_distributions(
    states: np.ndarray,
    model: nn.Module,
    action_dim: int,
    title: str = "Q-value distribution"
) -> None:
    """
    Plot histogram of Q-values for a batch of states.

    Args:
        states: np.ndarray of shape (batch_size, state_dim)
        model: Q-network
        action_dim: number of actions
        title: plot title
    """
    model.eval()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(states)
        q_vals = model(state_tensor)  # (batch, action_dim)
        q_vals_flat = q_vals.cpu().numpy().flatten()
        plt.hist(q_vals_flat, bins=30, alpha=0.7)
        plt.xlabel("Q-value")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.show()
    model.train()

# Example usage, after training any agent:
sample_states = np.random.randn(500, 8)  # or better: sample from the replay buffer
plot_q_distributions(sample_states, trained_policy_net, action_dim=4, title="Trained Double DQN Q-value Distribution")
```

---

### Project Exercise: Plot Side-by-Side Learning Curves

**Description:**
Aggregate the reward curves obtained after training DQN, Double DQN, and Dueling DQN agents, and plot them in the same graph for side-by-side comparison.

**Full code:**

```python
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_side_by_side_curves(
    curves: List[List[float]],
    labels: List[str],
    window: int = 10,
    title: str = "Learning Curves"
) -> None:
    plt.figure(figsize=(10,6))
    for curve, label in zip(curves, labels):
        if window > 1:
            rolling = np.convolve(curve, np.ones(window)/window, mode='valid')
            plt.plot(rolling, label=label)
        else:
            plt.plot(curve, label=label)
    plt.xlabel("Episode")
    plt.ylabel(f"Mean Reward (window={window})")
    plt.title(title)
    plt.legend()
    plt.show()

# Example usage:
# plot_side_by_side_curves([rewards_dqn, rewards_double, rewards_dueling], ["DQN", "Double DQN", "Dueling DQN"])
```

---

## Conclusion

You've learned how Double DQN reduces overestimation bias by splitting action selection and evaluation, and how Dueling DQN architectures represent value and advantage functions separately for more robust learning. With PyTorch code, you've extended vanilla DQN, built dueling architectures, trained the agents on LunarLander-v2, and visualized both learning progress and internal Q-value statistics.

**Next:** In the following posts, we'll introduce policy gradient methods and actor-critic algorithms—taking RL one step closer toward solving real-world sequential decision-making tasks!

---

**Try out the exercises with your own enhancements, and share your learning curves in the comments!**

# === USER ===

i am getting this error. what is the equivalent of "pip install" for "gymnasium[box2d]" but for uv?

```
gymnasium.error.DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`
```

# === ASSISTANT ===

Great question! To install the **Box2D** extras for Gymnasium using [uv](https://github.com/astral-sh/uv), you can use the same syntax as with pip, as **uv is a drop-in replacement for pip**:

```sh
uv pip install "gymnasium[box2d]"
```

However, **Box2D has a build-time dependency on `swig`**, which is a system-level package (not a Python package). You’ll need to install it separately, usually via your OS package manager.

**For Linux (Debian/Ubuntu):**
```sh
sudo apt-get install swig
```

**For macOS (Homebrew):**
```sh
brew install swig
```

**For Windows:**
- Download the [SWIG Windows installer](http://www.swig.org/download.html) and add swig to your PATH.

---

**Summary for uv:**
```sh
uv pip install "gymnasium[box2d]"
```

If you get a build error about SWIG, install swig with your OS package manager as above, then rerun the `uv pip install ...` command.

---

### Full setup example

```sh
# 1. Install SWIG for your system (see above)
# 2. Install Gymnasium with Box2D extras
uv pip install "gymnasium[box2d]"
```

That should resolve the error!

# === USER ===

it says "no matches found" for gymnasium[box2d]. how do i install it?

# === ASSISTANT ===

Great follow-up! The error “no matches found: gymnasium[box2d]” usually comes from the **shell** (like zsh) interpreting the square brackets as a pattern for filename expansion (called "globbing"). This is not a problem with `uv` or `pip` itself, but with the shell parsing the command.

### The Solution: Quote the requirement!

You need to **enclose the package name in quotes**—preferably **single quotes** in zsh or bash—to prevent the shell from interpreting the brackets:

```sh
uv pip install 'gymnasium[box2d]'
```
or
```sh
uv pip install "gymnasium[box2d]"
```

**If you're using zsh (where this error is most common), you must use quotes.**

> ℹ️ If you still get `no matches found` in zsh, make sure you're using single or double quotes (not backticks).

---

## Quick Recap

1. **Install SWIG:**
   - On Ubuntu/Debian: `sudo apt-get install swig`
   - On Mac: `brew install swig`
   - On Windows: [SWIG downloads & instructions](http://swig.org/download.html)
2. **Install Gymnasium Box2D extras:**
   ```sh
   uv pip install 'gymnasium[box2d]'
   ```

---

If you follow the above steps, the package and its dependencies should install properly!

# === USER ===

what is swig? installing that with homebrew did not help.

# === ASSISTANT ===

Great question! Let’s break it down:

---

## What is SWIG?

**SWIG** stands for **Simplified Wrapper and Interface Generator**.  
- It is a **tool that helps connect C/C++ code to other languages** (like Python, Java, etc.).
- In the Python ecosystem, certain packages with native C/C++ extensions (like Box2D) may need SWIG during installation to generate the Python wrappers for the C/C++ code.

---

## Do I always need SWIG?

- **If you're installing a Python package that distributes pre-built ("wheels") binaries (like most pip/uv installs on common platforms), you usually don’t need SWIG.**
- But, sometimes wheels are not available for your OS or Python version, so the package will attempt to build from source, requiring a compiler and tools like SWIG.

---

## Why did installing SWIG with Homebrew not help?

**Possible reasons:**
1. **Prebuilt wheels** for Box2D should usually avoid the need for SWIG. But if your environment (e.g., M1/M2 Mac, Python 3.12, or another non-standard setup) lacks a wheel, pip/uv falls back to building from source.
2. **Dependencies or other build tools** may also be missing (such as a C++ compiler or extra libraries).
3. **You may need to install Box2D separately** for Python if gymnasium isn't pulling in the correct dependency.
4. Sometimes, the [PyPI project gymnasium[box2d]](https://pypi.org/project/gymnasium/) depends on [`box2d-py`](https://pypi.org/project/box2d-py/), which is *not always actively maintained*.
5. **SWIG may not be in your `$PATH`** after installation via Homebrew. Try opening a new terminal.

---

## What to do next?

### 1️⃣ Try Installing the Correct Box2D Python Package Directly

Try installing [box2d-py](https://pypi.org/project/box2d-py/) **before** or **after** installing gymnasium:

```sh
uv pip install box2d-py
```

Or, for the more recently-maintained fork:

```sh
uv pip install Box2D
```

### 2️⃣ Install Gymnasium with Box2D (quoted)

```sh
uv pip install 'gymnasium[box2d]'
```

If there’s still an error, try explicitly upgrading pip (sometimes helps with binary wheels):

```sh
uv pip install --upgrade pip setuptools wheel
```

### 3️⃣ If you hit a build error for box2d-py with SWIG

You may see output like:
```
error: command 'swig' failed: No such file or directory
```
or
```
fatal error: Box2D/Box2D.h: No such file or directory
```

Double-check SWIG is available by running:
```sh
swig -version
```
If that fails, your shell does **not** see swig (try restarting your terminal or adding Homebrew to your PATH).

If it works, but you still can't build, you might be hitting a Box2D source build problem (common on Apple Silicon or latest Python). **The best workaround is to use the latest version of Python 3.11 or 3.10 for best compatibility.**

---

## TL;DR / Checklist

1. **Install Homebrew SWIG:**  
   `brew install swig`
2. **Install Box2D Python bindings:**  
   `uv pip install box2d-py` **or** `uv pip install Box2D`
3. **Install Gymnasium with Box2D extras:**  
   `uv pip install 'gymnasium[box2d]'`
4. **If you get persistent build errors:**  
   - Try a slightly older Python version (3.11 or 3.10) in a fresh virtual environment.
   - Consider using [Conda](https://conda.io/) which bundles C toolchains for you.
   - Try `pip install` instead of `uv pip install` to see if wheel is available.

---

### Summary Table

| Step                                 | Command                                    | Expectation                    |
|---------------------------------------|--------------------------------------------|--------------------------------|
| Install swig w/ Homebrew              | `brew install swig`                        | swig binary available          |
| Test swig availability                | `swig -version`                            | Shows version info             |
| Install Box2D Python bindings (wheel) | `uv pip install box2d-py` or `uv pip install Box2D` | Success or error              |
| Install gymnasium + Box2D extra       | `uv pip install 'gymnasium[box2d]'`        | Installs env and Box2D        |

---

### Still not working?

**Paste your install error message(s) here, and I’ll help you debug further!**  
It’s unfortunately a fiddly ecosystem at the moment, especially on brand new Python/Mac combos.

---

**References:**
- [Box2D-py PyPI](https://pypi.org/project/box2d-py/)
- [Box2D PyPI](https://pypi.org/project/Box2D/)
- [gymnasium[box2d] PyPI](https://pypi.org/project/gymnasium/)

---

Let me know your exact OS, Python version, and error message if you still have issues!

# === USER ===

i don't want to use gymnasium, or swig, or gyumnasium[box2d]. can we rewrite this blog post to not use that for any of the demos or exercises?

# === ASSISTANT ===

Absolutely! Let's rewrite the post to **not use Gymnasium or Box2D**, and not require swig. For deep RL, a good simple replacement is [the CartPole environment](https://www.gymlibrary.dev/environments/classic_control/cart_pole/), but since you don't want Gymnasium or anything dependent on it, we'll make our own **minimal game environment** for the RL agent.

For demonstration of Double DQN and Dueling DQN, we need:
- A simple environment that you can run without external dependencies (beyond numpy/torch/matplotlib)
- A working reward structure and and state/action interface
- The rest is PyTorch and pure Python, no gym needed!

We'll use a "LineWorld" (1D random walk to a goal), which is commonly used for tabular RL and is straightforward for Deep Q-learning.

---

+++
title = "Learn Reinforcement Learning with PyTorch, Part 5.4: Extensions—Double DQN and Dueling DQN (No Gym)"
author = "Artintellica"
date = "2024-06-16"
+++

# Learn Reinforcement Learning with PyTorch, Part 5.4: Extensions—Double DQN and Dueling DQN (No Gym)

## Introduction

Deep Q-Learning (DQN) and its extensions have revolutionized reinforcement learning. But vanilla DQN exhibits overestimation bias and doesn't effectively learn the distinction between state value and action advantage. **Double DQN** and **Dueling DQN** are two important improvements.

*In this post, you will learn about Double DQN and Dueling DQN, and implement both on a custom environment—no Gymnasium, Box2D, or swig required.*

---

## Mathematical Overview

### Q-Learning

Q-learning seeks the optimal action-value function $Q^*(s, a)$ via:

$$
Q^*(s, a) = \max_{\pi} \mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0 = s, a_0 = a\right]
$$

where $r_t$ is the reward at time $t$, $\gamma$ is the discount factor, and $\pi$ is a policy.

**DQN** uses neural networks to approximate $Q(s,a;\theta)$:

$$
y^{\text{DQN}} = r + \gamma \max_{a'} Q_{\text{target}}(s', a'; \theta^-)
$$

### Double DQN

DQN tends to overestimate action-values. **Double DQN** reduces this by separating action selection and evaluation:

$$
y^{\text{Double DQN}} = r + \gamma Q_{\text{target}}(s', \underset{a'}{\arg\max}\ Q_{\text{online}}(s', a'; \theta); \theta^-)
$$

### Dueling DQN

Dueling DQN splits Q-value estimation into state-value $V(s)$ and action-advantage $A(s,a)$:

$$
Q(s, a) = V(s) + \left(A(s,a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a')\right)
$$

---

## Connecting the Math to the Code

- **Double DQN:** When calculating the target, use the online network to pick the next action, but the target network to evaluate its Q-value.
- **Dueling DQN:** Architect the neural network such that it has separate output streams for $V(s)$ and $A(s,a)$, combined as above.

---

## Python Demos

### Demo 1: Minimal 1D "LineWorld" Environment

A simple environment: the agent starts in the middle, can move left or right, and receives +1 for reaching the rightmost cell (goal), -1 for reaching the leftmost cell (pit), 0 otherwise.

```python
import numpy as np
from typing import Tuple

class LineWorld:
    """
    A simple 1D walk world of N positions.
    The agent starts in the middle.
    Actions: 0 (left), 1 (right)
    Reward: +1 for rightmost, -1 for leftmost, 0 otherwise.
    """
    def __init__(self, size: int = 7):
        self.size = size
        self.start_pos = size // 2
        self.reset()

    def reset(self) -> np.ndarray:
        self.pos = self.start_pos
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        # One-hot position encoding
        obs = np.zeros(self.size, dtype=np.float32)
        obs[self.pos] = 1.0
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if action == 0:
            self.pos -= 1
        elif action == 1:
            self.pos += 1
        else:
            raise ValueError("Action must be 0 or 1")

        if self.pos <= 0:
            return self._get_obs(), -1.0, True
        elif self.pos >= self.size - 1:
            return self._get_obs(), 1.0, True
        else:
            return self._get_obs(), 0.0, False
```

---

## Exercises

### Exercise 1: Implement Double DQN Target Calculation

**Description:**  
Write the PyTorch function that, given batched rewards, dones, next states, and online/target networks, computes the batch of Double DQN targets for loss calculation.

```python
import torch
import torch.nn as nn

def compute_double_dqn_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_states: torch.Tensor,
    gamma: float,
    online_net: nn.Module,
    target_net: nn.Module
) -> torch.Tensor:
    """
    Compute Double DQN targets.
    """
    with torch.no_grad():
        next_q_online = online_net(next_states)  # (batch, n_actions)
        next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)  # (batch, 1)
        next_q_target = target_net(next_states)
        selected_q = torch.gather(next_q_target, 1, next_actions).squeeze(1)
        q_targets = rewards + gamma * selected_q * (1.0 - dones)
    return q_targets
```

---

### Exercise 2: Dueling DQN Model Implementation

**Description:**  
Write a simple dueling DQN model for LineWorld using fully typed PyTorch.

```python
import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals
```

---

### Exercise 3: Train DQN, Double DQN, and Dueling DQN on LineWorld

**Description:**  
Write a minimal DQN loop to compare regular DQN, Double DQN, and Dueling DQN. Use experience replay. Plot total reward per episode.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import List, Tuple

# Standard DQN model
class DQN(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Tuple[np.ndarray, int, float, np.ndarray, float]) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        samples = random.sample(self.buffer, batch_size)
        batch = list(zip(*samples))
        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.LongTensor(np.array(batch[1]))
        rewards = torch.FloatTensor(np.array(batch[2]))
        next_states = torch.FloatTensor(np.array(batch[3]))
        dones = torch.FloatTensor(np.array(batch[4]))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

def train_agent(
    model_type: str = "dqn",
    episodes: int = 300,
    buffer_size: int = 2000,
    hidden_dim: int = 64
) -> List[float]:
    env = LineWorld(size=7)
    state_dim = env.size
    action_dim = 2
    gamma = 0.97
    lr = 1e-3
    batch_size = 32
    start_train = 100
    target_update = 20

    if model_type == "dueling":
        online_net: nn.Module = DuelingDQN(state_dim, action_dim, hidden_dim)
        target_net: nn.Module = DuelingDQN(state_dim, action_dim, hidden_dim)
    else:
        online_net = DQN(state_dim, action_dim, hidden_dim)
        target_net = DQN(state_dim, action_dim, hidden_dim)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size)
    rewards_per_episode: List[float] = []
    steps = 0

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            epsilon = max(0.05, 0.9 - episode / 200)
            if np.random.rand() < epsilon:
                action = np.random.choice(action_dim)
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q_vals = online_net(state_t)
                action = int(torch.argmax(q_vals).item())

            next_state, reward, done = env.step(action)
            buffer.push((state, action, reward, next_state, float(done)))
            state = next_state
            total_reward += reward

            if len(buffer) >= start_train:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    if model_type == "double" or model_type == "dueling":
                        # Double DQN or Dueling DQN both use Double DQN targets
                        next_q_online = online_net(next_states)
                        next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
                        next_q_target = target_net(next_states)
                        next_q = next_q_target.gather(1, next_actions).squeeze(1)
                    else:
                        # Standard DQN
                        next_q = target_net(next_states).max(1)[0]
                    q_targets = rewards + gamma * next_q * (1.0 - dones)
                loss = nn.functional.mse_loss(q_values, q_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if steps % target_update == 0:
                    target_net.load_state_dict(online_net.state_dict())
                steps += 1

        rewards_per_episode.append(total_reward)

    return rewards_per_episode

# Train and plot
rewards_dqn = train_agent("dqn")
rewards_double = train_agent("double")
rewards_dueling = train_agent("dueling")

plt.plot(rewards_dqn, label="DQN")
plt.plot(rewards_double, label="Double DQN")
plt.plot(rewards_dueling, label="Dueling Double DQN")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.title("LineWorld: DQN vs Double DQN vs Dueling DQN")
plt.show()
```

---

### Exercise 4: Visualize Q-Value Distributions During Training

**Description:**  
Plot a histogram of all Q-values output by the trained Q-network on 50 random states.

```python
def plot_q_distribution(
    model: nn.Module,
    input_dim: int,
    action_dim: int,
    n_samples: int = 50
) -> None:
    states = np.eye(input_dim)  # all one-hot positions
    all_qs = []
    with torch.no_grad():
        for i in range(n_samples):
            state = states[np.random.randint(0, input_dim)]
            qvals = model(torch.FloatTensor(state).unsqueeze(0))
            all_qs += qvals.cpu().numpy().flatten().tolist()
    import matplotlib.pyplot as plt
    plt.hist(all_qs, bins=20, alpha=0.7)
    plt.xlabel("Q-value")
    plt.ylabel("Count")
    plt.title("Q-value Distribution")
    plt.show()
# To use after training:
# plot_q_distribution(your_trained_model, input_dim=7, action_dim=2)
```

---

### Project Exercise: Plot Side-by-Side Learning Curves

```python
def plot_learning_curves(
    curves: List[List[float]],
    labels: List[str],
    window: int = 10
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    for curve, label in zip(curves, labels):
        mean_curve = np.convolve(curve, np.ones(window) / window, mode='valid')
        plt.plot(mean_curve, label=label)
    plt.xlabel("Episode")
    plt.ylabel(f"Reward per Episode (window={window})")
    plt.title("Learning Curves LineWorld")
    plt.legend()
    plt.show()

# plot_learning_curves([rewards_dqn, rewards_double, rewards_dueling], ["DQN", "Double DQN", "Dueling DQN"])
```

---

## Conclusion

You’ve learned the mathematics and PyTorch implementation of **Double DQN** and **Dueling DQN**—100% free of Gymnasium, Box2D, and swig. With a minimal custom environment, you can now experiment and visualize the effects of these algorithms on a reinforcement learning problem, and adapt these ideas to more complex domains.

---

*Ready to extend to more complex environments, or perhaps craft your own? Try modifying `LineWorld` or building a discrete 2D grid! Share your results and questions below!*

# === USER ===


