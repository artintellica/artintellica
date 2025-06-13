import random


class TinyEnv:
    def __init__(self) -> None:
        self.state: int = 1  # start in middle
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


# Given (x, y) pairs, optimize for y
x_data = [0, 1, 2, 3]
y_data = [0, 1, 4, 9]


def supervised_train(x, y):
    model_output = [i**2 for i in x]
    loss = sum((y1 - y2) ** 2 for y1, y2 in zip(model_output, y))
    print("Supervised Loss:", loss)


supervised_train(x_data, y_data)

# Given only x, cluster/group/transform
x_data = [0.5, 1.5, 3.2, 8.1, 9.3]


def unsupervised_task(x):
    center = sum(x) / len(x)
    clusters = [0 if xi < center else 1 for xi in x]
    print("Unsupervised clusters:", clusters)


unsupervised_task(x_data)

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
