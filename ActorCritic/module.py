"""
@FileName：module.py
@Description：
@Author：ZuoChenyang
@Time：2023/8/10 14:18
@Copyright：©2023 ZuoChenyang
"""

import torch
import numpy as np


class PolicyNet(torch.nn.Module):
    """
    策略网络，用于输出策略信息
    输入为当前观测
    输出为动作的概率分布
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        return torch.nn.functional.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    """
    价值网络，用于输出对动作的价值判断
    输入为当前观测
    输出为对观测的价值评价
    """
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(x)


class ActorCriticAgent:
    def __init__(self, actor_network, critic_network, actor_optimizer, critic_optimizer, gamma=0.9,
                 device=torch.device("cpu")):
        self.actor_network = actor_network.to(device)
        self.critic_network = critic_network.to(device)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.gamma = gamma
        self.device = device

    def action(self, obs):
        obs = torch.tensor([obs], dtype=torch.float).to(self.device)
        probs = self.actor_network(obs)
        action_dict = torch.distributions.Categorical(probs)
        action = action_dict.sample()
        return action.item()

    def learning(self, transition_dict):
        states = torch.tensor(transition_dict["obs"], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict["next_obs"], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic_network(next_states) * (1-dones)
        td_delta = td_target - self.critic_network(states)
        temp1 = self.actor_network(states)
        temp2 = temp1.gather(1, actions)
        log_probs = torch.log(temp2)
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(torch.nn.functional.mse_loss(self.critic_network(states), td_target.detach()))

        self.actor_network.zero_grad()
        self.critic_network.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
