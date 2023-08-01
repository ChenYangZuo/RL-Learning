"""
@FileName：test2.py
@Description：
@Author：ZuoChenyang
@Time：2023/7/30 22:04
@Copyright：©2023 ZuoChenyang
"""
import numpy as np
import random

# 定义列表
numpy_list = np.array([0, 0, -1, 0])

# 找到所有最大值的索引
max_indices = np.where(numpy_list == np.max(numpy_list))[0]

# 在所有最大值中随机选择一个索引
random_max_index = random.choice(max_indices)

# 输出结果
print("随机选择的最大值索引:", random_max_index)
