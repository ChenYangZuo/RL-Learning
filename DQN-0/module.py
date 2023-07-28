import torch
import numpy as np


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
    def __init__(self, q_function, optimizer, n_action, gamma=0.9, e_greed=0.1):
        self.q_function = q_function
        self.optimizer = optimizer
        self.criterion = torch.nn.MSELoss()
        self.n_action = n_action
        self.gamma = gamma
        self.epsilon = e_greed

    # 神经网络输出最佳Action
    def predict(self, obs):
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
        pred_Vs = self.q_function(obs)
        predict_Q = pred_Vs[action]

        next_pred_Vs = self.q_function(next_obs)
        best_V = next_pred_Vs.max()
        target_Q = reward + (1 - float(done)) * self.gamma * best_V

        # 梯度下降更新参数
        self.optimizer.zero_grad()
        loss = self.criterion(predict_Q, target_Q)
        loss.backward()
        self.optimizer.step()
