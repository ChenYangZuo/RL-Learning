import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

import module

writer = SummaryWriter(log_dir='runs/train_data')


class TrainManager:
    def __init__(self, env, episodes=1000, lr=0.001, gamma=0.9, e_greed=0.1):
        self.env = env
        self.episodes = episodes

        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.n
        q_function = module.MLP(n_obs, n_act)

        optimizer = torch.optim.AdamW(q_function.parameters(), lr=lr)

        self.agent = module.DQNAgent(
            q_function=q_function,
            optimizer=optimizer,
            n_action=n_act,
            gamma=gamma,
            e_greed=e_greed
        )

    def train_episode(self):
        total_reward = 0
        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs)
        while True:
            action = self.agent.action(obs)
            next_obs, reward, done, _, _ = self.env.step(action)
            next_obs = torch.FloatTensor(next_obs)
            self.agent.learning(obs, action, reward, next_obs, done)
            total_reward += reward
            obs = next_obs
            if done:
                break
        return total_reward

    def test_episode(self):
        total_reward = 0
        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs)
        # obs = torch.tensor(obs)
        while True:
            action = self.agent.predict(obs)
            next_obs, reward, done, _, _ = self.env.step(action)
            next_obs = torch.FloatTensor(next_obs)
            total_reward += reward
            obs = next_obs
            # self.env.render()
            if done:
                break
        return total_reward

    def train(self):
        for e in range(self.episodes):
            episode_reward = self.train_episode()
            # print(f"Episode {e}: Reward={episode_reward}")
            # if e % 10 == 0:
            writer.add_scalar(
                tag="reward",  # 可以暂时理解为图像的名字
                scalar_value=episode_reward,  # 纵坐标的值
                global_step=e  # 当前是第几次迭代，可以理解为横坐标的值
            )

            if e % 100 == 0:
                test_reward = self.test_episode()
                print(f"Test Reward: {test_reward}")


if __name__ == '__main__':
    # env1 = gym.make("CartPole-v1", render_mode="human")
    env1 = gym.make("CartPole-v1")
    tm = TrainManager(
        env=env1
    )
    tm.train()

