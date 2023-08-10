"""
@FileName：train.py
@Description：
@Author：ZuoChenyang
@Time：2023/7/30 15:45
@Copyright：©2023 ZuoChenyang
"""

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import math

from env import FrozenLake_v2

import module

writer = SummaryWriter(log_dir='runs/train_data')


class TrainManager:
    def __init__(self, env, episodes=2000, alpha=0.1, gamma=0.9, e_greed=0.1, max_step=120):

        self.env = env
        self.episodes = episodes
        self.episodes_step = 0
        self.max_step = max_step

        n_obs = env.observation_space.n
        n_act = env.action_space.n

        self.agent = module.QLearning(
            epsilon=e_greed,
            alpha=alpha,
            gamma=gamma,
            n_state=n_obs,
            n_action=n_act
        )

    def train_episode(self, mode=0):
        total_reward = 0
        count = 0
        obs, _ = self.env.reset()
        for step in range(self.max_step):
            if mode == 0:
                action = self.agent.action(obs)
            elif mode == 2:
                action = self.agent.action_SA(obs)
            else:
                action = self.agent.action(obs)
            next_obs, reward, done, _, _ = self.env.step(action)
            self.agent.epsilon = 0.1 * math.exp(-1. * self.episodes_step / 2000)  # 动态修改探索率
            self.agent.learning(obs, action, reward, next_obs, mode)
            total_reward += reward
            obs = next_obs
            count = step
            self.episodes_step += 1
            if done:
                break
        self.agent.temperature *= 0.95
        return total_reward, count

    def test_episode(self):
        total_reward = 0
        obs, _ = self.env.reset()
        while True:
            print(obs)
            action = self.agent.predict(obs)
            next_obs, reward, done, _, _ = self.env.step(action)
            total_reward += reward
            obs = next_obs
            if done:
                break
        return total_reward

    def train(self, mode=0):
        for e in range(self.episodes):
            episode_reward, steps = self.train_episode(mode)
            print(f"{e} - {episode_reward}")
            if mode == 0:
                writer.add_scalar(
                    tag="reward_q",
                    scalar_value=episode_reward,
                    global_step=e
                )
                writer.add_scalar(
                    tag="steps_q",
                    scalar_value=steps,
                    global_step=e
                )
            elif mode == 1:
                writer.add_scalar(
                    tag="reward_ant",
                    scalar_value=episode_reward,
                    global_step=e
                )
                writer.add_scalar(
                    tag="steps_ant",
                    scalar_value=steps,
                    global_step=e
                )
            elif mode == 2:
                writer.add_scalar(
                    tag="reward_sa",
                    scalar_value=episode_reward,
                    global_step=e
                )
                writer.add_scalar(
                    tag="steps_sa",
                    scalar_value=steps,
                    global_step=e
                )
            # if e % 50 == 0:
            #     test_reward = self.test_episode()
            #     print(f"Test Reward: {test_reward}")


if __name__ == '__main__':
    # env1 = gym.make("CliffWalking-v0")
    # env1 = gym.make("FrozenLake8x8-v1", desc=None, map_name="8x8", is_slippery=False)
    env2 = FrozenLake_v2()
    env2.hole = [[0, 2], [0, 11], [0, 14], [0, 18], [1, 1], [1, 2], [1, 6], [1, 7], [1, 11], [1, 13], [1, 16], [2, 6],
                 [2, 14], [2, 15], [2, 16], [2, 17], [2, 18], [3, 4], [3, 9], [4, 6], [4, 16], [5, 3], [5, 8], [5, 15],
                 [5, 19], [6, 6], [6, 7], [6, 10], [6, 11], [7, 1], [7, 8], [7, 9], [7, 12], [7, 14], [7, 16], [7, 19],
                 [8, 6], [8, 12], [8, 13], [8, 16], [8, 19], [9, 1], [9, 14], [9, 15], [9, 17], [10, 0], [10, 6],
                 [10, 12], [10, 16], [11, 0], [11, 4], [11, 5], [11, 8], [11, 10], [11, 15], [12, 0], [12, 1], [12, 6],
                 [12, 9], [12, 10], [12, 12], [12, 15], [12, 17], [13, 6], [13, 10], [13, 14], [13, 18], [14, 1],
                 [14, 4], [14, 5], [14, 7], [14, 11], [14, 14], [14, 18], [14, 19], [15, 2], [15, 6], [15, 8], [16, 2],
                 [16, 4], [16, 8], [16, 11], [16, 12], [16, 14], [16, 16], [16, 17], [17, 2], [17, 4], [17, 12],
                 [17, 14], [18, 6], [18, 8], [18, 16], [18, 17], [18, 19], [19, 3], [19, 6], [19, 7], [19, 15]]
    env2.start = [19, 0]
    env2.end = [0, 19]

    tm = TrainManager(
        env=env2
    )
    tm.train(mode=0)
    print("-" * 20)
    # tm.test_episode()

    tm = TrainManager(
        env=env2
    )
    tm.train(mode=1)
    print("-" * 20)
    # tm.test_episode()

    tm = TrainManager(
        env=env2
    )
    tm.train(mode=2)
    print("-" * 20)
    # tm.test_episode()
