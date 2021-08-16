# -*- coding: utf-8 -*-
# ===========================================
# @Time    : 2021/8/16 上午10:46
# @Author  : shutao
# @FileName: t2q_train.py
# @remark  : 
# 
# @Software: PyCharm
# Github 　： https://github.com/NameLacker
# ===========================================

from models.t2p_model_stages import BiSeNet
from tools.t2p_reader import CityScapes
from models.loss import OhemCELoss, DetailAggregateLoss, boundary_loss
from tools.evaluation import evaluate

import paddle
import paddle.nn as nn

from paddle.io import DataLoader
from paddle.nn import functional as F

import os
import time
import os.path as osp
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--local_rank',
        dest='local_rank',
        type=int,
        default=-1,
    )
    parse.add_argument(
        '--n_workers_train',
        dest='n_workers_train',
        type=int,
        default=8,
    )
    parse.add_argument(
        '--n_workers_val',
        dest='n_workers_val',
        type=int,
        default=0,
    )
    parse.add_argument(
        '--n_img_per_gpu',
        dest='n_img_per_gpu',
        type=int,
        default=4,
    )
    parse.add_argument(
        '--max_iter',
        dest='max_iter',
        type=int,
        default=60000,
    )
    parse.add_argument(
        '--save_iter_sep',
        dest='save_iter_sep',
        type=int,
        default=1000,
    )
    parse.add_argument(
        '--warmup_steps',
        dest='warmup_steps',
        type=int,
        default=1000,
    )
    parse.add_argument(
        '--mode',
        dest='mode',
        type=str,
        default='train',
    )
    parse.add_argument(
        '--ckpt',
        dest='ckpt',
        type=str,
        default=None,
    )
    parse.add_argument(
        '--respath',
        dest='respath',
        type=str,
        default='/home/lacker/work/Dataset/cityscapes',
    )
    parse.add_argument(
        '--backbone',
        dest='backbone',
        type=str,
        default='STDCNet1446',
    )
    parse.add_argument(
        '--pretrain_path',
        dest='pretrain_path',
        type=str,
        default='',
    )
    parse.add_argument(
        '--use_conv_last',
        dest='use_conv_last',
        type=str2bool,
        default=False,
    )
    parse.add_argument(
        '--use_boundary_2',
        dest='use_boundary_2',
        type=str2bool,
        default=False,
    )
    parse.add_argument(
        '--use_boundary_4',
        dest='use_boundary_4',
        type=str2bool,
        default=False,
    )
    parse.add_argument(
        '--use_boundary_8',
        dest='use_boundary_8',
        type=str2bool,
        default=True,
    )
    parse.add_argument(
        '--use_boundary_16',
        dest='use_boundary_16',
        type=str2bool,
        default=False,
    )
    return parse.parse_args()


def train():
    args = parse_args()

    save_pth_path = os.path.join(args.respath, 'pths')
    dspth = '/home/lacker/work/Dataset/cityscapes'

    # print(save_pth_path)
    # print(osp.exists(save_pth_path))
    # if not osp.exists(save_pth_path) and dist.get_rank()==0:
    if not osp.exists(save_pth_path):
        os.makedirs(save_pth_path)

    # dataset
    n_classes = 19
    n_img_per_gpu = args.n_img_per_gpu
    n_workers_train = args.n_workers_train
    n_workers_val = args.n_workers_val
    use_boundary_16 = args.use_boundary_16
    use_boundary_8 = args.use_boundary_8
    use_boundary_4 = args.use_boundary_4
    use_boundary_2 = args.use_boundary_2

    mode = args.mode
    cropsize = [1024, 512]
    randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)

    # 数据集
    ds = CityScapes(dspth, cropsize=cropsize, mode='val', randomscale=randomscale)
    sampler = paddle.io.DistributedBatchSampler(ds, 4, shuffle=True, drop_last=True)
    dl = DataLoader(ds,
                    batch_sampler=sampler,
                    num_workers=1,
                    return_list=True)

    # 网络模型
    ignore_idx = 255
    net = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path,
                  use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, use_boundary_8=use_boundary_8,
                  use_boundary_16=use_boundary_16, use_conv_last=args.use_conv_last)
    net.train()

    # 损失函数
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_32 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    boundary_loss_func = DetailAggregateLoss()
    # boundary_loss_func = boundary_loss

    # optimizer
    maxmIOU50 = 0.
    maxmIOU75 = 0.
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = args.max_iter
    save_iter_sep = args.save_iter_sep
    power = 0.9
    warmup_steps = args.warmup_steps
    warmup_start_lr = 1e-5

    scheduler = paddle.optimizer.lr.PolynomialDecay(lr_start, decay_steps=max_iter, power=power)
    optim = paddle.optimizer.Momentum(
        learning_rate=scheduler,
        parameters=net.parameters(),
        weight_decay=weight_decay,
        momentum=momentum
    )

    # train loop
    msg_iter = 50
    loss_avg = []
    loss_boundery_bce = []
    loss_boundery_dice = []
    st = glob_st = time.time()
    epoch = 0
    iters = 0
    while iters < max_iter:
        for batch_id, (im, lb) in enumerate(dl):
            iters += 1
            H, W = im.shape[2:]
            lb = paddle.squeeze(lb, 1)

            optim.clear_gradients()
            boundary_loss_func.clear_gradients()

            if use_boundary_2 and use_boundary_4 and use_boundary_8:
                out, out16, out32, detail2, detail4, detail8 = net(im)

            if (not use_boundary_2) and use_boundary_4 and use_boundary_8:
                out, out16, out32, detail4, detail8 = net(im)

            if (not use_boundary_2) and (not use_boundary_4) and use_boundary_8:
                out, out16, out32, detail8 = net(im)

            if (not use_boundary_2) and (not use_boundary_4) and (not use_boundary_8):
                out, out16, out32 = net(im)

            lossp = criteria_p(out, lb)
            loss2 = criteria_16(out16, lb)
            loss3 = criteria_32(out32, lb)

            boundery_bce_loss = 0.
            boundery_dice_loss = 0.

            if use_boundary_2:
                # if dist.get_rank()==0:
                #     print('use_boundary_2')
                boundery_bce_loss2, boundery_dice_loss2 = boundary_loss_func(detail2, lb)
                boundery_bce_loss += boundery_bce_loss2
                boundery_dice_loss += boundery_dice_loss2

            if use_boundary_4:
                # if dist.get_rank()==0:
                #     print('use_boundary_4')
                boundery_bce_loss4, boundery_dice_loss4 = boundary_loss_func(detail4, lb)
                boundery_bce_loss += boundery_bce_loss4
                boundery_dice_loss += boundery_dice_loss4

            if use_boundary_8:
                # if dist.get_rank()==0:
                #     print('use_boundary_8')
                boundery_bce_loss8, boundery_dice_loss8 = boundary_loss_func(detail8, lb)
                boundery_bce_loss += boundery_bce_loss8
                boundery_dice_loss += boundery_dice_loss8

            loss = lossp + loss2 + loss3 + boundery_bce_loss + boundery_dice_loss

            loss.backward()
            optim.step()
            print(loss.numpy()[0], boundery_bce_loss.numpy()[0], boundery_dice_loss.numpy()[0], optim.get_lr())

            if iters == max_iter:
                print("结束训练")
                exit(0)


if __name__ == '__main__':
    train()
