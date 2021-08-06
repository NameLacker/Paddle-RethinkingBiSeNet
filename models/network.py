# -*- coding: utf-8 -*-
"""
========================================
@Time   ：2021/8/6 17:16
@Auther ：shutao
@File   ：network.py
@IDE    ：PyCharm
@Github ：https://github.com/NameLacker
@Gitee  ：https://gitee.com/nameLacker
========================================
"""

import paddle

from paddle import nn


class BiSeNet(nn.Layer):
    def __init__(self):
        super(BiSeNet, self).__init__()

    def forward(self, inputs):
        return inputs
