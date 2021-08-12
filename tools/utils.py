# -*- coding: utf-8 -*-
"""
========================================
@Time   ：2021/8/6 17:16
@Auther ：shutao
@File   ：utils.py
@IDE    ：PyCharm
@Github ：https://github.com/NameLacker
@Gitee  ：https://gitee.com/nameLacker
========================================
"""

import numpy as np
import yaml
import os


def get_configuration():
    """
    获取项目配置
    :return:
    """
    # 获取yaml文件路径
    yamlPath = "config.yaml"
    # open方法打开直接读出来
    f = open(yamlPath, 'r', encoding='utf-8')
    cfg = f.read()
    d = yaml.load(cfg)  # 用load方法转字典
    return d


def fill_ndarray(t):
    """
    替换nan值为平均值
    :param t: 
    :return: 
    """
    nan_num = np.count_nonzero(t != t)  # 判断该列存在不为0的数个数
    if nan_num != 0:
        temp_not_nan_col = t[t == t]
        t[np.isnan(t)] = temp_not_nan_col.mean()
    return np.mean(t)
