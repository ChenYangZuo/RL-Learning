# Readme

设计REINFORCE Without Baseline框架

## Environment

- Python 3.10
- Pytorch 1.17.1 With CUDA11.7
- gymnasium 0.28.1
- numpy 1.25.0
- pygame 2.1.3
- matplotlib 3.7.1
- tensorboard 2.10.0


## module.py

使用Pytorch构建全连接层神经网络拟合Policy Function

基于神经网络构建REINFORCE智能体，通过梯度下降优化神经网络

## train.py

构建了TrainManager管理训练过程

据悉自GYM0.18.0后，环境的reward_threshold并未真正实装，环境不会因步数和累计奖励置位Done信号，因此需要手动判定累计奖励，否则算法不收敛