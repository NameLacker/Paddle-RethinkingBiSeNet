# -*- coding: utf-8 -*-
# ===========================================
# @Time    : 2021/8/16 下午4:18
# @Author  : shutao
# @FileName: t2p_evaluation.py
# @remark  : 
# 
# @Software: PyCharm
# Github 　： https://github.com/NameLacker
# ===========================================

from paddle.io import DataLoader
import paddle.nn.functional as F
import paddle

import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import math


class MscEvalV0(object):
    def __init__(self, scale=0.5, ignore_label=255):
        self.ignore_label = ignore_label
        self.scale = scale

    def __call__(self, net, dl, n_classes):
        hist = paddle.zeros((n_classes, n_classes))
        for i, (imgs, label) in enumerate(dl):
            N, C, H, W = label.shape

            label = label.squeeze(1)
            size = label.shape[-2:]

            N, C, H, W = imgs.shape
            new_hw = [int(H * self.scale), int(W * self.scale)]

            imgs = F.interpolate(imgs, size=new_hw, mode='bilinear', align_corners=True)

            logits = net(imgs)

            logits = F.interpolate(logits, size=size,
                                   mode='bilinear', align_corners=True)
            probs = F.softmax(logits, axis=1)
            preds = paddle.argmax(probs, axis=1)
            keep = label != self.ignore_label
            hist += paddle.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
            ).flatten(n_classes, n_classes).float()
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item()
