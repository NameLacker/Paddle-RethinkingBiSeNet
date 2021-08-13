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
import cv2 as cv
import paddle

import paddle.nn as nn
import paddle.nn.functional as F


def dice_loss_func(inputs, target):
    smooth = 1.
    iflat = inputs.flatten()
    tflat = target.flatten()
    intersection = paddle.sum(iflat * tflat)
    loss = 1 - (2.*intersection + smooth)/(paddle.sum(paddle.square(iflat)) + paddle.sum(paddle.square(tflat)) + smooth)
    return loss


def boundary_loss(boundary_logits, gtmasks):
    """
    DetailAggregate 损失函数
    :param boundary_logits:
    :param gtmasks:
    :return:
    """
    laplacian_kernel = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=np.float32).reshape((3, 3))
    fuse_kernel = np.array([[6. / 10], [3. / 10], [1. / 10]], dtype=np.float32).reshape(3)
    gtmasks = gtmasks.numpy().astype(np.uint8)

    boundary_targets_pyramids = np.zeros_like(gtmasks)
    for idx, gtmask in enumerate(gtmasks):
        gtmask = cv.filter2D(gtmask, -1, kernel=laplacian_kernel)

        gtmask_x2 = cv.resize(gtmask, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST).clip(min=0)
        gtmask_x4 = cv.resize(gtmask, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST).clip(min=0)

        gtmask_x2_up = cv.resize(gtmask_x2, gtmasks.shape[1:][::-1], interpolation=cv.INTER_NEAREST)
        gtmask_x4_up = cv.resize(gtmask_x4, gtmasks.shape[1:][::-1], interpolation=cv.INTER_NEAREST)

        gtmask[gtmask > 0.1] = 1
        gtmask[gtmask <= 0.1] = 0

        gtmask_x2_up[gtmask_x2_up > 0.1] = 1
        gtmask_x2_up[gtmask_x2_up <= 0.1] = 0

        gtmask_x4_up[gtmask_x4_up > 0.1] = 1
        gtmask_x4_up[gtmask_x4_up <= 0.1] = 0

        boundary_targets_pyramid = gtmask*fuse_kernel[0] + gtmask_x2_up*fuse_kernel[1] + gtmask_x4_up*fuse_kernel[2]
        boundary_targets_pyramid[boundary_targets_pyramid > 0.1] = 1
        boundary_targets_pyramid[boundary_targets_pyramid <= 0.1] = 0

        boundary_targets_pyramids[idx] = boundary_targets_pyramid
    boundary_targets_pyramids = paddle.to_tensor(boundary_targets_pyramids, dtype=paddle.float32).unsqueeze(1)

    bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets_pyramids)
    dice_loss = dice_loss_func(F.sigmoid(boundary_logits), boundary_targets_pyramids)
    return bce_loss, dice_loss


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
        """
        todo: 用paddle实现的此方法会导致在训练阶段显存利用率持续增长
        """
        super(DetailAggregateLoss, self).__init__()

        self.laplacian_kernel = paddle.to_tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=paddle.float32).reshape((1, 1, 3, 3))
        # self.laplacian_kernel = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=np.float32).reshape((3, 3))

        self.fuse_kernel = paddle.to_tensor([[6. / 10], [3. / 10], [1. / 10]],
                                            dtype=paddle.float32).reshape((1, 3, 1, 1))
        # self.fuse_kernel = np.array([[6./10], [3./10], [1./10]], dtype=np.float32).reshape((1, 3, 1, 1))

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

        boundary_targets_pyramid = paddle.unsqueeze(gtmasks, axis=1).astype(paddle.float32)
        boundary_targets_pyramid[boundary_targets_pyramid > 0.1] = 1
        boundary_targets_pyramid[boundary_targets_pyramid <= 0.1] = 0

        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets_pyramid)
        dice_loss = dice_loss_func(F.sigmoid(boundary_logits), boundary_targets_pyramid)
        return bce_loss, dice_loss
