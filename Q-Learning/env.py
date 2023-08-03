"""
@FileName：env.py
@Description：
@Author：ZuoChenyang
@Time：2023/8/1 11:06
@Copyright：©2023 ZuoChenyang
"""

import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces

import numpy as np
import random
from copy import deepcopy


def array_to_int(state, size=7):
    return state[0] * size + state[1]


class FrozenLake_v2:
    def __init__(self):
        self.rows = 8  # 8行
        self.cols = 8  # 8列
        self.start = [0, 0]
        self.end = [7, 7]
        # 左行右列
        self.hole = [[2, 3], [3, 5], [4, 3], [5, 1], [5, 2], [6, 1], [6, 4], [6, 6], [7, 3]]
        self.current_state = self.start
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.rows * self.cols)

    def step(self, action):
        new_state = deepcopy(self.current_state)
        if action == 0:  # Move left
            new_state[1] = max(self.current_state[1] - 1, 0)
        elif action == 1:  # Move down
            new_state[0] = min(self.current_state[0] + 1, self.rows - 1)
        elif action == 2:  # Move right
            new_state[1] = min(self.current_state[1] + 1, self.cols - 1)
        elif action == 3:  # Move up
            new_state[0] = max(self.current_state[0] - 1, 0)
        else:
            raise Exception("Invalid action")
        self.current_state = new_state

        if new_state in self.hole:  # 掉到洞里
            done = True
            reward = -100
        elif new_state == self.end:  # 到达终点
            done = True
            reward = 100
        else:  # 正常走一步
            done = False
            reward = -1

        return array_to_int(self.current_state, self.cols), reward, done, 0, 0

    def render(self):
        pass

    def reset(self):
        self.current_state = self.start
        return 0, 0
