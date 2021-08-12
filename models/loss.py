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


def dice_loss_func(inputs, target):
    smooth = 1.
    iflat = inputs.flatten()
    tflat = target.flatten()
    intersection = paddle.sum(iflat * tflat)
    loss = 1 - ((2. * intersection + smooth) / (paddle.sum(iflat) + paddle.sum(tflat) + smooth))
    return loss


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

        self.laplacian_kernel = paddle.to_tensor(
            [-1, -1, -1, -1, -8, -1, -1, -1, -1],
            dtype=paddle.float32).reshape((1, 1, 3, 3))

        x = paddle.to_tensor([[6./10], [3./10], [1./10]], dtype=paddle.float32).reshape((1, 3, 1, 1))
        self.fuse_kernel = paddle.create_parameter(x.shape, dtype=str(x.numpy().dtype),
                                                   default_initializer=paddle.nn.initializer.Assign(x))

    def forward(self, boundary_logits, gtmasks):
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).astype(paddle.float32), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clip(min=0)

        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).astype(paddle.float32), self.laplacian_kernel,
                                       stride=2, padding=1).clip(min=0)
        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).astype(paddle.float32), self.laplacian_kernel,
                                       stride=4, padding=1).clip(min=0)
        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).astype(paddle.float32), self.laplacian_kernel,
                                       stride=8, padding=1).clip(min=0)

        boundary_targets_x2_up = F.upsample(boundary_targets_x2, size=boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.upsample(boundary_targets_x4, size=boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x8_up = F.upsample(boundary_targets_x8, size=boundary_targets.shape[2:], mode='nearest')

        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0

        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0

        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0

        boundary_targets_pyramids = paddle.stack([boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up],
                                                 axis=1).squeeze(2)

        boundary_targets_pyramid = F.conv2d(boundary_targets_pyramids, self.fuse_kernel)
        boundary_targets_pyramid[boundary_targets_pyramid > 0.1] = 1
        boundary_targets_pyramid[boundary_targets_pyramid <= 0.1] = 0

        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.upsample(boundary_logits, size=boundary_targets.shape[2:], mode='bilinear',
                                         align_corners=True)
        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets_pyramid)
        dice_loss = dice_loss_func(F.sigmoid(boundary_logits), boundary_targets_pyramid)
        return bce_loss, dice_loss
