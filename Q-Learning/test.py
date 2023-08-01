"""
@FileName：test.py
@Description：
@Author：ZuoChenyang
@Time：2023/7/30 17:21
@Copyright：©2023 ZuoChenyang
"""

import gymnasium as gym
import random
import numpy as np

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)

state_number = env.observation_space
action_number = env.action_space
print(state_number)
print(action_number)
test1 = [2, 2, 1, 1, 1, 2]
test2 = [2, 2, 2, 1]
obs, _ = env.reset()
print(obs)
print("-"*20)
for i in test1:
    next_obs, reward, done, _, _ = env.step(i)
    print(next_obs)
    print(reward)
    print(done)
    print("-"*20)

obs, _ = env.reset()
print(obs)
print("-"*20)
for i in test2:
    next_obs, reward, done, _, _ = env.step(i)
    print(next_obs)
    print(reward)
    print(done)
    print("-"*20)