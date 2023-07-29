import torch
import numpy as np
import copy

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
    def __init__(self, q_function, optimizer, replay_buffer, replay_start_size, batch_size, update_period, n_action, gamma=0.9,
                 e_greed=0.1, device=torch.device("cpu")):
        self.predict_network = q_function.to(device)
        self.target_network = copy.deepcopy(q_function).to(device)
        self.update_period = update_period

        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        self.criterion = torch.nn.MSELoss()
        self.n_action = n_action
        self.gamma = gamma
        self.epsilon = e_greed
        self.device = device

        self.global_step = 0

    # 神经网络输出最佳Action
    def predict(self, obs):
        # obs = torch.FloatTensor(obs)
        # Q_list = self.predict_network(obs)
        # action = int(torch.argmax(Q_list).detach().numpy())
        state = torch.tensor(np.array([obs]), dtype=torch.float).to(self.device)
        action = self.predict_network(state).argmax().item()
        return action

    # 考虑探索时输出最终Action
    def action(self, obs):
        # 几率探索
        if np.random.uniform(0, 1) < self.epsilon:
            # action = np.random.choice(self.n_action)
            action = np.random.randint(self.n_action)
        else:
            action = self.predict(obs)
        return action

    # 执行Action后迭代优化神经网络
    def learning(self, obs, action, reward, next_obs, done):
        # 向经验池中新增数据
        self.replay_buffer.append((obs, action, reward, next_obs, done))
        self.global_step += 1
        # 当经验池中数据多于指定值时开始训练
        # 每隔固定轮数后训练一次预测网络
        if len(self.replay_buffer) > self.replay_start_size and self.global_step % self.replay_buffer.step == 0:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = self.replay_buffer.sample(self.batch_size)
            transition_dict = {
                "batch_obs": batch_obs,
                "batch_action": batch_action,
                "batch_reward": batch_reward,
                "batch_next_obs": batch_next_obs,
                "batch_done": batch_done
            }
            self.batch_learning(transition_dict)
        # 每隔固定轮数后更新一次目标网络
        if self.global_step % self.update_period == 0:
            self.sync_target()

    # 从经验池中获取一个batch的数据进行训练
    def batch_learning(self, transition_dict):
        batch_obs = torch.FloatTensor(transition_dict["batch_obs"]).to(self.device)
        batch_action = torch.tensor(transition_dict["batch_action"], dtype=torch.int64).view(-1, 1).to(self.device)
        batch_reward = torch.FloatTensor(transition_dict["batch_reward"]).to(self.device)
        batch_next_obs = torch.FloatTensor(transition_dict["batch_next_obs"]).to(self.device)
        batch_done = torch.tensor(transition_dict['batch_done'], dtype=torch.float).view(-1, 1).to(self.device)

        # Only-in CPU
        # predict_Vs = self.predict_network(batch_obs)  # Tensor(32,2)
        # action_one_hot = utils.one_hot(batch_action, self.n_action).to(self.device)
        # predict_Q = (predict_Vs * action_one_hot).sum(dim=1).to(self.device)
        predict_Q = self.predict_network(batch_obs).gather(1, batch_action)

        # next_predict_Vs = self.target_network(batch_next_obs)
        # best_V = next_predict_Vs.max(1)[0].view(-1, 1)  # DQN做法
        max_action = self.predict_network(batch_next_obs).max(1)[1].view(-1, 1)
        best_V = self.target_network(batch_next_obs).gather(1, max_action)

        target_Q = batch_reward + (1 - batch_done) * self.gamma * best_V

        # 梯度下降更新参数
        self.optimizer.zero_grad()
        loss = self.criterion(predict_Q, target_Q)
        loss.backward()
        self.optimizer.step()

    # 将预测网络的参数更新至目标网络
    def sync_target(self):
        # for target_param, param in zip(self.target_network.parameters(), self.predict_network.parameters()):
        #     target_param.data.copy_(param.data)
        self.target_network.load_state_dict(self.predict_network.state_dict())
