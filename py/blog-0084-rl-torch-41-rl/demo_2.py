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
