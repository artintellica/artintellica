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
        self.state_space = [
            (x, y)
            for x in range(width)
            for y in range(height)
            if (x, y) not in self.walls
        ]

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
        grid = np.full((self.height, self.width), " ")
        for x, y in self.walls:
            grid[y, x] = "#"
        for x, y in self.pits:
            grid[y, x] = "P"
        gx, gy = self.goal
        grid[gy, gx] = "G"
        sx, sy = self.start
        grid[sy, sx] = "S"
        ax, ay = self.agent_pos
        grid[ay, ax] = "A"
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                c = grid[y, x]
                if policy is not None and c == " ":
                    s_idx = self.state_space.index((x, y))
                    a = np.argmax(policy[s_idx])
                    a_char = "↑→↓←"[a]
                    line += a_char
                else:
                    line += c
            print(line)
        print("")

    def state_to_idx(self, state: Tuple[int, int]) -> int:
        return self.state_space.index(state)

    def idx_to_state(self, idx: int) -> Tuple[int, int]:
        return self.state_space[idx]
