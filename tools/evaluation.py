# -*- coding: utf-8 -*-
# ===========================================
# @Time    : 2021/8/12 上午10:59
# @Author  : shutao
# @FileName: evaluation.py
# @remark  : 
# 
# @Software: PyCharm
# Github 　： https://github.com/NameLacker
# ===========================================

import numpy as np
import paddle

import paddle.nn.functional as F


def evaluate(image, label, net, n_classes=19, scale=0.5, ignore_label=255):
    size = label.shape[-2:]
    N, C, H, W = image.shape
    new_hw = [int(H * scale), int(W * scale)]
    image = F.upsample(image, size=new_hw, mode="bilinear", align_corners=True)  # STDC2-Se50标准的预测输入

    logits = net(image)

    logits = F.upsample(logits, size=size, mode="bilinear", align_corners=True)
    probs = F.softmax(logits, axis=1)
    preds = paddle.argmax(probs, axis=1)
    keep = label != ignore_label

    np_keep = keep.numpy()
    np_label = label.numpy()
    np_preds = preds.numpy()
    hist = np.bincount(np_label[np_keep] * n_classes + np_preds[np_keep],
                       minlength=n_classes ** 2).reshape((n_classes, n_classes)).astype(np.float32)
    return hist
