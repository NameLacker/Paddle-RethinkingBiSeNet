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

import numpy as np
import logging
import paddle
import time
import os

from paddle import optimizer
import paddle.nn.functional as F
from visualdl import LogWriter

from tools.data_reader import CityScapes
from models.network import BiSeNet
from models.loss import OhemCELoss, boundary_loss
from tools.utils import get_configuration
from tools.evaluation import evaluate

cfg = get_configuration()
log_name = str(int(time.time()))
log_writer = LogWriter("log/train_" + log_name)
logger = logging.getLogger()


def init_log_config():
    """
    初始化日志相关配置
    :return:
    """
    global logger
    # 设置日志级别
    logger.setLevel(logging.DEBUG)  # 设置日志级别
    # 日志文件保存配置
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, 'train_' + str(int(time.time())) + '.log')
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 日志控制台输出配置
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    # 添加配置
    logger.addHandler(streamHandler)
    logger.addHandler(fh)


init_log_config()  # 初始化日志输出


def run_train():
    # 获取训练配置
    img_size = 224
    train_cfg = cfg["train_config"]
    if not os.path.exists(train_cfg["params_dir"]):
        os.mkdir(train_cfg["params_dir"])
    n_classes = train_cfg["n_classes"]
    use_boundary_2 = train_cfg["use_boundary_2"]
    use_boundary_4 = train_cfg["use_boundary_4"]
    use_boundary_8 = train_cfg["use_boundary_8"]
    continue_train = train_cfg["continue_train"]
    opt_config = train_cfg["opt_config"]  # 读取优化器配置
    load_params = train_cfg["load_params"]

    train_dataset = CityScapes(datalist_file="data/val.list")
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=train_cfg["batch_size"], shuffle=True, drop_last=True)
    train_loader = paddle.io.DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=0,
                                        return_list=True)
    eval_dataset = CityScapes(is_test=True)
    eval_batch_sampler = paddle.io.DistributedBatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    eval_loader = paddle.io.DataLoader(eval_dataset, batch_sampler=eval_batch_sampler, num_workers=0, return_list=True)

    net = BiSeNet(num_classes=n_classes,
                  use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, use_boundary_8=use_boundary_8)

    if continue_train:
        logger.info("加载模型参数...")
        model_static = paddle.load(load_params)
        net.load_dict(model_static)

    # 优化器配置
    values = [value * opt_config["learning_rate"] for value in opt_config["values"]]
    scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=opt_config["boundaries"],
                                                   values=values)
    opt = optimizer.SGD(learning_rate=scheduler, parameters=net.parameters(), weight_decay=5e-4)

    # 损失函数
    score_thres = 0.7
    n_min = 16 * img_size * img_size // 16
    criteria_loss = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=255)

    maxmIOU50 = -1
    stop_count = 0
    train_step = 0
    val_step = 0
    for epoch_id in range(train_cfg["num_epochs"]):
        net.train()
        for batch_id, (image, label) in enumerate(train_loader):
            if use_boundary_2 and use_boundary_4 and use_boundary_8:
                out, out16, out32, detail2, detail4, detail8 = net(image)
            elif (not use_boundary_2) and use_boundary_4 and use_boundary_8:
                out, out16, out32, detail4, detail8 = net(image)
            elif (not use_boundary_2) and (not use_boundary_4) and use_boundary_8:
                out, out16, out32, detail8 = net(image)
            else:
                out, out16, out32 = net(image)
            # 分割损失
            lossp = criteria_loss(out, label)
            loss2 = criteria_loss(out16, label)
            loss3 = criteria_loss(out32, label)

            boundery_bce_loss, boundery_dice_loss = 0, 0
            if use_boundary_2:
                # detail2 损失
                boundery_bce_loss2, boundery_dice_loss2 = boundary_loss(detail2, label)
                boundery_bce_loss += boundery_bce_loss2
                boundery_dice_loss += boundery_dice_loss2

            if use_boundary_4:
                boundery_bce_loss4, boundery_dice_loss4 = boundary_loss(detail4, label)
                boundery_bce_loss += boundery_bce_loss4
                boundery_dice_loss += boundery_dice_loss4

            if use_boundary_8:
                # detail8 损失
                boundery_bce_loss8, boundery_dice_loss8 = boundary_loss(detail8, label)
                boundery_bce_loss += boundery_bce_loss8
                boundery_dice_loss += boundery_dice_loss8

            loss = lossp + loss2 + loss3 + boundery_bce_loss + boundery_dice_loss
            loss.backward()
            opt.minimize(loss)
            net.clear_gradients()
            log_writer.add_scalar("train/loss", step=train_step, value=loss.numpy()[0])
            log_writer.add_scalar("train/boundery_bce_loss", step=train_step, value=boundery_bce_loss.numpy()[0])
            log_writer.add_scalar("train/boundery_dice_loss", step=train_step, value=boundery_dice_loss.numpy()[0])
            train_step += 1
            logger.info("Epoch: {}, Batch: {}, "
                        "Loss: {:.4}, boundery_bce_loss: {:.4}, boundery_dice_loss: {:.4}, "
                        "learning_rate: {}"
                        .format(epoch_id, batch_id,
                                loss.numpy()[0], boundery_bce_loss.numpy()[0], boundery_dice_loss.numpy()[0],
                                scheduler.get_lr()))

        net.eval()
        num_eval = eval_dataset.__len__()
        hist = np.zeros((n_classes, n_classes), dtype=np.float32)
        for batch_id, (image, label) in enumerate(eval_loader):
            hist += evaluate(image, label, net, n_classes=n_classes)
            print("\r", end="")
            percentage = int(100 * batch_id / num_eval)
            print("Evalution progress rate: {}%: ".format(percentage), "▋" * (percentage // 2), end="")
        print()
        ious = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mIOU50 = np.nanmean(ious)  # 增加对有nan值的处理
        log_writer.add_scalar("eval/mIOU50", step=val_step, value=mIOU50)
        val_step += 1

        logger.info("当前验证集平均 mIOU50: {:.4}".format(mIOU50))
        if mIOU50 >= maxmIOU50:
            stop_count = 0
            maxmIOU50 = mIOU50
            save_pth = os.path.join(train_cfg["params_dir"], "model_maxmIOU50_{:.2}.pdparams".format(maxmIOU50 * 100))
            model = net.state_dict()
            paddle.save(model, save_pth)
            logger.info("成功保存模型参数...")
        else:
            stop_count += 1
        if stop_count == 5:
            logger.info("提前停止训练")
            exit(0)
        scheduler.step()  # 更新学习率


if __name__ == '__main__':
    run_train()
