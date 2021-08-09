# -*- coding: utf-8 -*-
"""
========================================
@Time   ：2021/8/6 17:14
@Auther ：shutao
@File   ：train.py
@IDE    ：PyCharm
@Github ：https://github.com/NameLacker
@Gitee  ：https://gitee.com/nameLacker
========================================
"""

import paddle

from paddle import optimizer
from paddle.io import DataLoader

from tools.data_reader import SegDataset
from models.network import BiSeNet

from tools.utils import get_configuration

cfg = get_configuration()


def run_train():
    # 获取训练配置
    train_cfg = cfg["train_config"]

    train_dataset = SegDataset()
    train_reader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    test_dataset = SegDataset(is_test=True)
    test_reader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    net = BiSeNet()

    opt = optimizer.SGD(learning_rate=train_cfg["learning_rate"], parameters=net.parameters())

    for epoch_id in range(train_cfg["num_epochs"]):
        for batch_id, (image, label) in enumerate(train_reader):
            pred = net(image)
            pass
    pass


if __name__ == '__main__':
    run_train()
