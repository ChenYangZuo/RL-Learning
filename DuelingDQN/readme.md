# Readme

设计Dueling DQN框架，应用Target Network与Double DQN

## Environment

- Python 3.10
- Pytorch 1.17.1 With CUDA11.7
- gymnasium 0.28.1
- numpy 1.25.0
- pygame 2.1.3
- matplotlib 3.7.1
- tensorboard 2.10.0

## replaybuffer.py

创建了经验回放池，支持一致性采样

## module.py

使用Pytorch构建单全连接层神经网络，由V与A构成

基于神经网络构建DQN智能体，支持epsilon探索环境，通过梯度下降优化神经网络

## train.py

构建了TrainManager管理训练过程