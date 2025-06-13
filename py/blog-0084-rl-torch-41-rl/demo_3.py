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
