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
