# -*- encoding: utf-8 -*-
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
from models.loss import OhemCELoss, DetailAggregateLoss

from tools.utils import get_configuration

cfg = get_configuration()


def run_train():
    # 获取训练配置
    img_size = 224
    train_cfg = cfg["train_config"]

    train_dataset = SegDataset()
    train_reader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    test_dataset = SegDataset(is_test=True)
    test_reader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    net = BiSeNet()

    score_thres = 0.7
    n_min = 16 * img_size * img_size // 16
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=255)
    boundary_loss_func = DetailAggregateLoss()

    opt = optimizer.SGD(learning_rate=train_cfg["learning_rate"], parameters=net.parameters())

    for epoch_id in range(train_cfg["num_epochs"]):
        net.train()
        for batch_id, (image, label) in enumerate(train_reader):
            stage6, lp_seg_out, lp_detail_out = net(image)
            seg_loss = criteria_p(lp_seg_out, label)
            detail_loss = boundary_loss_func(lp_detail_out, label)

        net.eval()
        for batch_id, (image, label) in enumerate(test_reader):
            pred = net(image)
        pass


if __name__ == '__main__':
    run_train()
