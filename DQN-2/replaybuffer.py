"""
@FileName：replaybuffer.py
@Description：
@Author：ZuoChenyang
@Time：2023/7/28 10:16
@Copyright：©2023 ZuoChenyang
"""

import collections
import random

from torch import FloatTensor
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, step):
        self.buffer = collections.deque(maxlen=max_size)  # FIFO结构
        self.step = step

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)  # 一致性采样
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*mini_batch)
        obs_batch = FloatTensor(np.array(obs_batch))
        action_batch = FloatTensor(np.array(action_batch))
        reward_batch = FloatTensor(np.array(reward_batch))
        next_obs_batch = FloatTensor(np.array(next_obs_batch))
        done_batch = FloatTensor(np.array(done_batch))
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self):
        return len(self.buffer)
