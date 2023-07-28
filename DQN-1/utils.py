"""
@FileName：utils.py
@Description：
@Author：ZuoChenyang
@Time：2023/7/28 10:35
@Copyright：©2023 ZuoChenyang
"""

import torch


def one_hot(index_list, class_num):
    """
    可以将索引变为one_hot形式的张量
    :param index_list: [2,1,3,0]
    :param class_num: 类别数量
    :return:
    tensor([[0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.]])
    """
    if type(index_list) == torch.Tensor:
        index_list = index_list.detach().numpy()
    indexes = torch.LongTensor(index_list).view(-1, 1)
    out = torch.zeros(len(index_list), class_num)
    out = out.scatter_(dim=1, index=indexes, value=1)
    return out
