"""
@FileName：train.py
@Description：
@Author：ZuoChenyang
@Time：2023/8/10 14:37
@Copyright：©2023 ZuoChenyang
"""

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

import module

writer = SummaryWriter(log_dir='runs/train_data')


class TrainManager:
    def __init__(self, env, episodes=1000, actor_lr=0.001, critic_lr=0.001, gamma=0.9, cuda=True):
        self.env = env
        self.episodes = episodes
        if cuda:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.n
        actor_network = module.PolicyNet(n_obs, 256, n_act)
        critic_network = module.ValueNet(n_obs, 256)

        actor_optimizer = torch.optim.AdamW(actor_network.parameters(), lr=actor_lr)
        critic_optimizer = torch.optim.AdamW(critic_network.parameters(), lr=critic_lr)

        self.agent = module.ActorCriticAgent(
            actor_network=actor_network,
            critic_network=critic_network,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            gamma=gamma,
            device=self.device
        )

    def train_episode(self):
        total_reward = 0
        transition_dict = {
            'obs': [],
            'actions': [],
            'next_obs': [],
            'rewards': [],
            'dones': []
        }
        obs, _ = self.env.reset()
        while True:
            action = self.agent.action(obs)
            next_obs, reward, done, _, _ = self.env.step(action)
            if total_reward == 200:
                done = True
            transition_dict['obs'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_obs'].append(next_obs)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            total_reward += reward
            obs = next_obs
            if done:
                break
        self.agent.learning(transition_dict)
        return total_reward

    def train(self):
        for e in range(self.episodes):
            episode_reward = self.train_episode()
            writer.add_scalar(
                tag="reward",  # 可以暂时理解为图像的名字
                scalar_value=episode_reward,  # 纵坐标的值
                global_step=e  # 当前是第几次迭代，可以理解为横坐标的值
            )
            if e % 100 == 0:
                print(f"{e} Reward: {episode_reward}")


if __name__ == '__main__':
    env1 = gym.make("CartPole-v1", render_mode="rgb_array")
    tm = TrainManager(
        env=env1,
        actor_lr=1e-3,
        critic_lr=1e-2,
        gamma=0.98
    )
    tm.train()