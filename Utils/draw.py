"""
@FileName：draw.py
@Description：
@Author：ZuoChenyang
@Time：2023/8/6 14:58
@Copyright：©2023 ZuoChenyang
"""

import matplotlib.pyplot as plt
import numpy as np
import csv


def smooth_data(data, smooth):
    if smooth >= 1 or smooth < 0:
        raise ValueError("The smooth parameter must be greater than 0 and not greater than 1")
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * smooth + (1 - smooth) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_single_data(data):
    if 'data_x' not in data or 'data_y' not in data or 'label' not in data:
        raise ValueError("Invalid data format. Input dictionary should contain 'data_x', 'data_y', and 'label' keys.")
    plt.plot(data['data_x'], data['data_y'], label=data['label'])


def draw_linechart(data, label_x, label_y, title):
    if isinstance(data, list):  # 输入是一族数据
        for d in data:
            plot_single_data(d)
        plt.legend()  # 输入数据族时自动显示图例
    elif isinstance(data, dict):  # 输入是单条数据
        plot_single_data(data)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    q_y_data = []
    ant_y_data = []
    sa_y_data = []

    with open(r'D:\迅雷下载\q.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # 读取并忽略第一行
        column_index = header.index('Value')  # 找到列名所在的索引
        for row in reader:
            column_data = row[column_index]
            q_y_data.append(float(column_data))

    with open(r'D:\迅雷下载\ant.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # 读取并忽略第一行
        column_index = header.index('Value')  # 找到列名所在的索引
        for row in reader:
            column_data = row[column_index]
            ant_y_data.append(float(column_data))

    with open(r'D:\迅雷下载\sa.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # 读取并忽略第一行
        column_index = header.index('Value')  # 找到列名所在的索引
        for row in reader:
            column_data = row[column_index]
            sa_y_data.append(float(column_data))

    x_data = list(range(len(q_y_data)))

    data_smooth = 0.6

    list_data = [{
        'data_x': x_data,
        'data_y': smooth_data(q_y_data, data_smooth),
        'label': 'Q-Learning'
    }, {
        'data_x': x_data,
        'data_y': smooth_data(ant_y_data, data_smooth),
        'label': 'Q-Learning With Ant'
    }, {
        'data_x': x_data,
        'data_y': smooth_data(sa_y_data, data_smooth),
        'label': 'Q-Learning With SA'
    }]
    draw_linechart(list_data, "x", "y", "test")

