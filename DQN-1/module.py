import torch
import numpy as np

import utils


# 多层感知机
class MLP(torch.nn.Module):

    def __init__(self, obs_size, n_action):
        super().__init__()
        self.mlp = self.__mlp(obs_size, n_action)

    def __mlp(self, obs_size, n_action):
        return torch.nn.Sequential(
            torch.nn.Linear(obs_size, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, n_action)
        )

    def forward(self, x):
        return self.mlp(x)


# DQN智能体
class DQNAgent(object):
    def __init__(self, q_function, optimizer, replay_buffer, replay_start_size, batch_size, n_action, gamma=0.9,
                 e_greed=0.1):
        self.q_function = q_function
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        self.criterion = torch.nn.MSELoss()
        self.n_action = n_action
        self.gamma = gamma
        self.epsilon = e_greed
        self.global_step = 0

    # 神经网络输出最佳Action
    def predict(self, obs):
        obs = torch.FloatTensor(obs)
        Q_list = self.q_function(obs)
        action = int(torch.argmax(Q_list).detach().numpy())
        return action

    # 考虑探索时输出最终Action
    def action(self, obs):
        # 几率探索
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_action)
        else:
            action = self.predict(obs)
        return action

    # 执行Action后迭代优化神经网络
    def learning(self, obs, action, reward, next_obs, done):
        self.replay_buffer.append((obs, action, reward, next_obs, done))
        self.global_step += 1
        if len(self.replay_buffer) > self.replay_start_size and self.global_step % self.replay_buffer.step == 0:
            self.batch_learning(*self.replay_buffer.sample(self.batch_size))

    def batch_learning(self, batch_obs, batch_action, batch_reward, batch_next_obs, batch_done):
        pred_Vs = self.q_function(batch_obs)
        action_onehot = utils.one_hot(batch_action, self.n_action)
        predict_Q = (pred_Vs * action_onehot).sum(dim=1)

        next_pred_Vs = self.q_function(batch_next_obs)
        best_V = next_pred_Vs.max(1)[0]
        target_Q = batch_reward + (1 - batch_done) * self.gamma * best_V

        # 梯度下降更新参数
        self.optimizer.zero_grad()
        loss = self.criterion(predict_Q, target_Q)
        loss.backward()
        self.optimizer.step()
