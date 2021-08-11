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

from tools.data_reader import CityScapes
from models.network import BiSeNet
from models.loss import OhemCELoss, DetailAggregateLoss

from tools.utils import get_configuration

cfg = get_configuration()


def run_train():
    # 获取训练配置
    img_size = 224
    train_cfg = cfg["train_config"]
    use_boundary_2 = train_cfg["use_boundary_2"]
    use_boundary_4 = train_cfg["use_boundary_4"]
    use_boundary_8 = train_cfg["use_boundary_8"]

    train_dataset = CityScapes(datalist_file="data/val.list")
    train_reader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=False)
    test_dataset = CityScapes(is_test=True)
    test_reader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    net = BiSeNet(use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, use_boundary_8=use_boundary_8)

    score_thres = 0.7
    n_min = 16 * img_size * img_size // 16
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=255)
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=255)
    criteria_32 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=255)
    boundary_loss_func = DetailAggregateLoss()

    # 融合网络和损失的参数，便于同时更新两者的参数
    net_parameters = net.parameters()
    # net_parameters.extend(boundary_loss_func.parameters())
    opt = optimizer.SGD(learning_rate=train_cfg["learning_rate"], parameters=net_parameters)

    for epoch_id in range(train_cfg["num_epochs"]):
        net.train()
        for batch_id, (image, label) in enumerate(train_reader):
            if use_boundary_2 and use_boundary_4 and use_boundary_8:
                out, out16, out32, detail2, detail4, detail8 = net(image)
            elif (not use_boundary_2) and use_boundary_4 and use_boundary_8:
                out, out16, out32, detail4, detail8 = net(image)
            elif (not use_boundary_2) and (not use_boundary_4) and use_boundary_8:
                out, out16, out32, detail8 = net(image)
            else:
                out, out16, out32 = net(image)
            # 分割损失
            lossp = criteria_p(out, label)
            loss2 = criteria_16(out16, label)
            loss3 = criteria_32(out32, label)

            boundery_bce_loss, boundery_dice_loss = 0., 0.
            if use_boundary_2:
                # detail2 损失
                boundery_bce_loss2,  boundery_dice_loss2 = boundary_loss_func(detail2, label)
                boundery_bce_loss += boundery_bce_loss2
                boundery_dice_loss += boundery_dice_loss2

            if use_boundary_4:
                boundery_bce_loss4,  boundery_dice_loss4 = boundary_loss_func(detail4, label)
                boundery_bce_loss += boundery_bce_loss4
                boundery_dice_loss += boundery_dice_loss4

            if use_boundary_8:
                # detail8 损失
                boundery_bce_loss8,  boundery_dice_loss8 = boundary_loss_func(detail8, label)
                boundery_bce_loss += boundery_bce_loss8
                boundery_dice_loss += boundery_dice_loss8

            loss = lossp + loss2 + loss3 + boundery_bce_loss + boundery_dice_loss
            loss.backward()
            opt.minimize(loss)
            net.clear_gradients()
            print("Epoch: {}, Batch: {}, Loss: {}, learning_rate: None".format(epoch_id, batch_id, loss.numpy()[0]))

        net.eval()
        for batch_id, (image, label) in enumerate(test_reader):
            pred = net(image)
        pass


if __name__ == '__main__':
    run_train()
