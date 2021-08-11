# -*- encoding: utf-8 -*-
# ===========================================
# @Time    : 2021/8/10 上午9:21
# @Author  : shutao
# @FileName: loss.py
# @remark  : 
# 
# @Software: PyCharm
# Github 　： https://github.com/NameLacker
# ===========================================

import numpy as np
import paddle

import paddle.nn as nn
import paddle.nn.functional as F


class OhemCELoss(nn.Layer):
    def __init__(self, thresh, n_min, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -np.log(thresh)
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none', axis=1)

    def forward(self, logits, labels):
        N, C, H, W = logits.shape
        loss = self.criteria(logits, labels).flatten()
        loss = paddle.sort(loss, descending=True)
        loss_np = loss.numpy()  # 辅助定位
        if loss[self.n_min] > self.thresh:
            loss_np = loss_np[loss_np > self.thresh]
            loss = loss[:loss_np.size]
        else:
            loss = loss[:self.n_min]
        return paddle.mean(loss)


class DetailAggregateLoss(nn.Layer):
    def __init__(self):
        super(DetailAggregateLoss, self).__init__()
        self.conv = nn.Conv2D(1, 1, 3)

    def forward(self, boundary_logits, gtmasks):
        loss = self.conv(boundary_logits)
        return loss
