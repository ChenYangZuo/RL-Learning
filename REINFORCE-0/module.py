"""
@FileName：module.py
@Description：
@Author：ZuoChenyang
@Time：2023/7/30 13:43
@Copyright：©2023 ZuoChenyang
"""

import torch
import numpy as np


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        return torch.nn.functional.softmax(self.fc2(x), dim=1)


class REINFORCEAgent(object):
    def __init__(self, v_function, optimizer, n_action, gamma=0.9, device=torch.device("cpu")):
        self.policy_network = v_function.to(device)
        self.optimizer = optimizer

        self.n_action = n_action
        self.gamma = gamma
        self.device = device

    # 输出最终Action
    def action(self, obs):
        obs = torch.tensor([obs], dtype=torch.float).to(self.device)
        probs = self.policy_network(obs)
        action_dict = torch.distributions.Categorical(probs)
        action = action_dict.sample()
        return action.item()

    # 执行Action后迭代优化神经网络
    def learning(self, transition_dict):
        reward_list = transition_dict["rewards"]
        state_list = transition_dict["obs"]
        action_list = transition_dict["actions"]

        g_at = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_network(state).gather(1, action))
            g_at = g_at * self.gamma + reward
            loss = -log_prob * g_at
            loss.backward()
        self.optimizer.step()
