# Readme

实现了最简单的DNQ

## Environment

- Python 3.10
- Pytorch 1.17.1 With CUDA11.7
- gymnasium 0.28.1
- numpy 1.25.0
- pygame 2.1.3
- matplotlib 3.7.1
- tensorboard 2.10.0

## module.py

使用Pytorch构建MLP多层感知机

基于MLP构建DQN智能体，支持epsilon探索环境，通过梯度下降优化神经网络

## train.py

构建了TrainManager管理训练过程