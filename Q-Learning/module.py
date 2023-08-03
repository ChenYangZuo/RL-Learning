"""
@FileName：module.py
@Description：
@Author：ZuoChenyang
@Time：2023/7/30 15:43
@Copyright：©2023 ZuoChenyang
"""

import numpy as np
import random


class QLearning:
    """ Q-learning算法 """
    def __init__(self, epsilon, alpha, gamma, n_state, n_action):
        self.Q_table = np.zeros([n_state, n_action])  # 初始化Q(s,a)表格
        self.tau_table = np.zeros([n_state, n_state])  # 初始化tau(s,s)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 贪婪策略中的参数
        random.seed(0)

    def predict(self, state):
        # action = np.argmax(self.Q_table[state])
        # 找到所有最大值的索引
        max_indices = np.where(self.Q_table[state] == np.max(self.Q_table[state]))[0]
        # 在所有最大值中随机选择一个索引
        action = random.choice(max_indices)
        return action

    def action(self, state):  # 贪婪选取下一步的操作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = self.predict(state)
        return action

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def learning(self, s0, a0, r, s1, mode=0):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        if mode != 0:  # 信息素机制
            self.tau_table[s0, s1] += r / 99
            self.Q_table[s0, a0] += self.alpha * (td_error + self.tau_table[s0, s1])
        else:  # 原始Q-Learning
            self.Q_table[s0, a0] += self.alpha * td_error
