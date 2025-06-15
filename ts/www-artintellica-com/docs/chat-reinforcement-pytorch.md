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

please produce the blog post for section 5.2, whose description is:

### 2. Deep Q-Networks (DQN): Concepts and PyTorch Implementation

* **Exercise 1:** Build a Q-network using `nn.Module` for CartPole-v1.
* **Exercise 2:** Implement an experience replay buffer from scratch.
* **Exercise 3:** Train DQN agent and plot episode rewards.
* **Exercise 4:** Save and load trained DQN models.

  * *Project Exercise:* Experiment with different network sizes and plot learning curves.

please respond with the blog post only.

# === ASSISTANT ===

+++
title = "Learn Reinforcement Learning with PyTorch, Part 5.2: Deep Q-Networks (DQN): Concepts and PyTorch Implementation"
author = "Artintellica"
date = "2024-06-15"
+++

# Learn Reinforcement Learning with PyTorch, Part 5.2: Deep Q-Networks (DQN): Concepts and PyTorch Implementation

---

## Introduction

So far in our series, we’ve explored reinforcement learning (RL) using tabular methods—explicitly storing and updating a value for every state–action pair. But the real world is far more complex: many problems have huge or continuous state spaces where tabular methods simply cannot scale. Enter Deep Q-Networks (DQN), the game-changing approach that brought deep learning to RL. This post will introduce you to DQN concepts and walk you through a full working PyTorch implementation, using OpenAI Gym’s classic **CartPole-v1** environment as our running example.

By the end, you’ll have built a neural Q-network, implemented experience replay, trained your own agent, and learned to tune and save your models. All code is hands-on, fully typed, and annotated for understanding.

---

## Mathematical Concepts Overview

Let’s review and extend the math underlying DQN.

### 1. The Q-Function and Bellman Equation

Recall that the **action-value function** (Q-function) for a state–action pair $(s, a)$ under policy $\pi$, is:

$$
Q^\pi(s, a) = \mathbb{E}\left[ G_t \mid s_t = s,\, a_t = a,\, \pi \right]
$$

where
- $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$
- $\gamma$ is the discount factor, $0 \leq \gamma < 1$

**The Bellman optimality equation for Q* (the optimal Q-function):**

$$
Q^*(s, a) = \mathbb{E}_{s'} \left[ r + \gamma \max_{a'} Q^*(s', a')\ \Big|\, s, a \right]
$$

### 2. Deep Q-Networks

In DQN, we **approximate** $Q^*(s,a)$ with a neural network parameterized by $\theta$. The network takes an observation (the state) as input and outputs Q-values for every possible action.

- **Input:** state $s$
- **Output:** $Q(s, a; \theta), \ \forall a \in \mathcal{A}$

The goal: minimize the **mean squared error (MSE)** between the current Q-value and the Bellman target for experience $(s, a, r, s', done)$:

$$
\text{Target:}\quad y = 
\begin{cases}
  r & \text{if done (terminal)}\\
  r + \gamma \max_{a'} Q(s', a';\ \theta^{-}) & \text{if not done}
\end{cases}
$$

where $\theta^{-}$ are the parameters of a "target network," usually a slowly updated snapshot of the main Q-network.

**DQN Learning Objective:**

$$
L(\theta) = \mathbb{E}_{(s, a, r, s', done) \sim \mathcal{D}} \left[
    \Big(Q(s,a;\theta) - y\Big)^2
\right]
$$

where $\mathcal{D}$ is a **replay buffer** (experience memory).

### 3. Experience Replay

To break correlations and improve sample efficiency, DQN samples batches **randomly** from a replay buffer of recent experiences instead of updating from only the latest transition.

### 4. Target Networks

A separate, periodically updated copy of the Q-network ($\theta^-$) is used to compute targets $y$, stabilizing training by reducing target oscillations.

---

## From Math to Code: Building DQN Step by Step

1. **Q-Network:** A standard MLP (multi-layer perceptron) that estimates $Q(s,a)$ from input states.
2. **Replay Buffer:** Stores $(s,a,r,s',done)$ transitions; sample minibatches for updates.
3. **Training Loop:**
    - For each time step:
        - Select action (epsilon-greedy: random with probability $\epsilon$, else greedy)
        - Step environment, collect $(s,a,r,s',done)$, add to replay buffer
        - Sample batch, calculate targets, calculate MSE loss, backpropagate
    - Occasionally update target network

---

## Python Demos

Let's walk through each core DQN component, with explanations and type hints.

---

### Demo 1: Build a Q-network (`nn.Module`) for CartPole-v1

```python
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)  # Shape: [batch, action_dim]
```

---

### Demo 2: Implement an Experience Replay Buffer

```python
import random
from typing import List, Tuple
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)
```

---

### Demo 3: Complete DQN Agent and Training Loop for CartPole-v1

```python
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

# Hyperparameters
GAMMA: float = 0.99
BATCH_SIZE: int = 64
REPLAY_SIZE: int = 10000
LEARNING_RATE: float = 1e-3
TARGET_UPDATE_FREQ: int = 100
EPS_START: float = 1.0
EPS_END: float = 0.01
EPS_DECAY: int = 500  # steps

# Initialize environment and networks
env = gym.make("CartPole-v1")
state_dim: int = env.observation_space.shape[0]
action_dim: int = env.action_space.n

q_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer(REPLAY_SIZE)

# Epsilon-greedy policy
def select_action(state: np.ndarray, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            q_values = q_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

# Training loop
num_episodes: int = 300
episode_rewards: List[float] = []
steps_done: int = 0

for episode in range(num_episodes):
    state, _ = env.reset(seed=None)
    episode_reward: float = 0.0
    done = False
    while not done:
        epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * steps_done / EPS_DECAY)
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        steps_done += 1

        # Train step
        if len(replay_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
            states_t = torch.from_numpy(states).float()
            actions_t = torch.from_numpy(actions).long().unsqueeze(1)
            rewards_t = torch.from_numpy(rewards).float().unsqueeze(1)
            next_states_t = torch.from_numpy(next_states).float()
            dones_t = torch.from_numpy(dones.astype(np.float32)).float().unsqueeze(1)

            # Q(s_t, a)
            q_values = q_net(states_t).gather(1, actions_t)
            # Compute target Q-values
            with torch.no_grad():
                max_next_q = target_net(next_states_t).max(1, keepdim=True)[0]
                targets = rewards_t + GAMMA * max_next_q * (1 - dones_t)

            # Loss
            loss = F.mse_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target net
        if steps_done % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())

    episode_rewards.append(episode_reward)
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

env.close()

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training on CartPole-v1")
plt.show()
```

---

### Demo 4: Save and Load a Trained DQN Model

```python
# Save model weights
torch.save(q_net.state_dict(), "dqn_cartpole.pth")

# To load:
q_net_loaded = QNetwork(state_dim, action_dim)
q_net_loaded.load_state_dict(torch.load("dqn_cartpole.pth", map_location=torch.device("cpu")))
q_net_loaded.eval()
```

---

## Python Exercises

Let's deepen your understanding with concrete coding challenges.

---

### **Exercise 1 — Build a Q-network using `nn.Module` for CartPole-v1**

_Define a class `QNetwork` as above, but allow the number of hidden layers and units per layer to be specified dynamically._

```python
from typing import List, Tuple
import torch
from torch import nn, Tensor

class CustomQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        dims = [state_dim] + hidden_dims
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1]) for i in range(len(hidden_dims))]
        )
        self.out = nn.Linear(dims[-1], action_dim)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.out(x)
```

---

### **Exercise 2 — Implement an experience replay buffer from scratch**

_Implement a class `ReplayBuffer` as shown, but add a method `.sample_indices(batch_size)` that returns both the sampled experiences and their indices in the buffer (useful for prioritized replay)._

```python
import numpy as np
import random
from typing import List, Tuple

class ReplayBufferWithIndices:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample_indices(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]
    ]:
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones, indices

    def __len__(self) -> int:
        return len(self.buffer)
```

---

### **Exercise 3 — Train a DQN agent and plot episode rewards**

_Use your Q-network and replay buffer to train an agent on CartPole-v1. Keep track of rewards per episode and plot them with Matplotlib._

```python
import gymnasium as gym
import torch
import numpy as np
from typing import List
import matplotlib.pyplot as plt

# Define network, buffer, hyperparameters
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = CustomQNetwork(state_dim, action_dim, hidden_dims=[128, 128])
target_net = CustomQNetwork(state_dim, action_dim, hidden_dims=[128, 128])
target_net.load_state_dict(q_net.state_dict())
target_net.eval()
optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
buffer = ReplayBufferWithIndices(5000)

BATCH_SIZE = 64
GAMMA = 0.99
TARGET_FREQ = 50
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 750

def select_action(state: np.ndarray, step: int) -> int:
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-step/EPS_DECAY)
    if np.random.rand() < eps:
        return np.random.randint(action_dim)
    with torch.no_grad():
        s = torch.from_numpy(state).float().unsqueeze(0)
        return int(q_net(s).argmax(1).item())

rewards_per_episode: List[float] = []
total_steps = 0
for episode in range(200):
    state, _ = env.reset()
    done = False
    ep_rew = 0.0
    while not done:
        action = select_action(state, total_steps)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        ep_rew += reward
        total_steps += 1

        if len(buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones, _ = buffer.sample_indices(BATCH_SIZE)
            s = torch.from_numpy(states).float()
            a = torch.from_numpy(actions).long().unsqueeze(1)
            r = torch.from_numpy(rewards).float().unsqueeze(1)
            ns = torch.from_numpy(next_states).float()
            d = torch.from_numpy(dones.astype(np.float32)).unsqueeze(1)
            q = q_net(s).gather(1, a)
            with torch.no_grad():
                q_next = target_net(ns).max(1, keepdim=True)[0]
                y = r + GAMMA * q_next * (1 - d)
            loss = torch.nn.functional.mse_loss(q, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if total_steps % TARGET_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())
    rewards_per_episode.append(ep_rew)

plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Rewards per Episode")
plt.show()
env.close()
```

---

### **Exercise 4 — Save and load trained DQN models**

_Use `torch.save` and `torch.load` to serialize and deserialize your trained DQN model's parameters._

```python
# Saving
torch.save(q_net.state_dict(), "custom_dqn.pth")

# Loading
loaded_net = CustomQNetwork(state_dim, action_dim, hidden_dims=[128, 128])
loaded_net.load_state_dict(torch.load("custom_dqn.pth"))
loaded_net.eval()
```

---

### **Project Exercise — Experiment with different network sizes and plot curves**

_Train and compare DQNs with one vs. two hidden layers, 64 vs. 256 units. Plot learning curves for all four variants on the same chart._

```python
configs = [
    ([64], "1x64"),
    ([256], "1x256"),
    ([64, 64], "2x64"),
    ([256, 256], "2x256")
]
env = gym.make("CartPole-v1")
all_rewards: List[List[float]] = []

for hidden_layers, label in configs:
    q_net = CustomQNetwork(state_dim, action_dim, hidden_dims=hidden_layers)
    target_net = CustomQNetwork(state_dim, action_dim, hidden_dims=hidden_layers)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = ReplayBufferWithIndices(5000)
    rewards = []
    total_steps = 0
    for episode in range(100):
        state, _ = env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            eps = EPS_END + (EPS_START-EPS_END)*np.exp(-total_steps/EPS_DECAY)
            action = np.random.randint(action_dim) if np.random.rand() < eps else \
                int(q_net(torch.from_numpy(state).float().unsqueeze(0)).argmax(1).item())
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_rew += reward
            total_steps += 1
            if len(buffer) >= BATCH_SIZE:
                s, a, r, ns, d, _ = buffer.sample_indices(BATCH_SIZE)
                s = torch.from_numpy(s).float()
                a = torch.from_numpy(a).long().unsqueeze(1)
                r = torch.from_numpy(r).float().unsqueeze(1)
                ns = torch.from_numpy(ns).float()
                d = torch.from_numpy(d.astype(np.float32)).unsqueeze(1)
                q = q_net(s).gather(1, a)
                with torch.no_grad():
                    q_next = target_net(ns).max(1, keepdim=True)[0]
                    y = r + GAMMA * q_next * (1 - d)
                loss = torch.nn.functional.mse_loss(q, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if total_steps % TARGET_FREQ == 0:
                target_net.load_state_dict(q_net.state_dict())
        rewards.append(ep_rew)
    all_rewards.append((label, rewards))

# Plotting
plt.figure(figsize=(10,6))
for label, rewards in all_rewards:
    plt.plot(rewards, label=label)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.title("DQN: Effect of Network Size on CartPole-v1")
plt.show()
env.close()
```

---

## Conclusion

Deep Q-Networks combine the strong function-approximation capability of neural networks with classic Q-learning, enabling RL agents to handle large/continuous state spaces. The essential tricks—experience replay, target networks—are all now in your hands, and you’ve coded a fully working DQN for the classic CartPole problem.

Don’t forget: you can always tweak the architecture, batch size, epsilon schedule and more to see how your agent learns. In the next lesson, we’ll push DQN further with modern extensions like Double and Dueling DQN. Happy experimenting!

# === USER ===


