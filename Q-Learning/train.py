"""
@FileName：train.py
@Description：
@Author：ZuoChenyang
@Time：2023/7/30 15:45
@Copyright：©2023 ZuoChenyang
"""

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter\

from env import FrozenLake_v2

import module

writer = SummaryWriter(log_dir='runs/train_data')


class TrainManager:
    def __init__(self, env, episodes=1000, alpha=0.1, gamma=0.9, e_greed=0.1):

        self.env = env
        self.episodes = episodes

        n_obs = env.observation_space.n
        n_act = env.action_space.n

        self.agent = module.QLearning(
            epsilon=e_greed,
            alpha=alpha,
            gamma=gamma,
            n_state=n_obs,
            n_action=n_act
        )

    def train_episode(self):
        total_reward = 0
        obs, _ = self.env.reset()
        for step in range(99):
            # steps.append(obs)
            action = self.agent.action(obs)
            # actions.append(action)
            next_obs, reward, done, _, _ = self.env.step(action)
            self.agent.learning(obs, action, reward, next_obs)
            total_reward += reward
            obs = next_obs
            # if reward == -100:
            #     done = True
            if done:
                break
        return total_reward

    def test_episode(self):
        total_reward = 0
        obs, _ = self.env.reset()
        while True:
            print(obs)
            action = self.agent.predict(obs)
            next_obs, reward, done, _, _ = self.env.step(action)
            total_reward += reward
            obs = next_obs
            # if reward == -100:
            #     done = True
            if done:
                break
        return total_reward

    def train(self):
        for e in range(self.episodes):
            episode_reward = self.train_episode()
            print(f"{e} - {episode_reward}")
            writer.add_scalar(
                tag="reward",
                scalar_value=episode_reward,
                global_step=e
            )
            # if e % 50 == 0:
            #     test_reward = self.test_episode()
            #     print(f"Test Reward: {test_reward}")


if __name__ == '__main__':
    # env1 = gym.make("CliffWalking-v0")
    # env1 = gym.make("FrozenLake8x8-v1", desc=None, map_name="8x8", is_slippery=False)
    env2 = FrozenLake_v2()
    tm = TrainManager(
        env=env2
    )
    tm.train()
    print("-"*20)
    tm.test_episode()
