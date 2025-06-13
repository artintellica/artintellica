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
        if action == 0:
            self.state -= 1
        else:
            self.state += 1
        if self.state <= 0:
            self.state = 0
            self.done = True
            return self.state, -1.0, True
        if self.state >= 2:
            self.state = 2
            self.done = True
            return self.state, 1.0, True
        return self.state, 0.0, False


# Exercise 1/2
env = MiniEnv()
state = env.reset()
for t in range(8):
    action = 1 if t % 2 == 0 else 0  # alternate
    next_state, reward, done = env.step(action)
    print(f"t={t}, state={next_state}, reward={reward}, done={done}")
    if done:
        break

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
        if done:
            break
    ep_rewards.append(total)
plt.plot(ep_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Random Policy, Per-Episode Reward")
plt.show()
