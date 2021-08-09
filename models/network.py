# -*- coding: utf-8 -*-
"""
========================================
@Time   ：2021/8/6 17:16
@Auther ：shutao
@File   ：network.py
@IDE    ：PyCharm
@Github ：https://github.com/NameLacker
@Gitee  ：https://gitee.com/nameLacker
========================================
"""

import paddle

from paddle import nn
from paddle.nn import functional as F

from .modules import Conv2D, STDC, FFM, ContextPath, BisNetOutput


class BiSeNet(nn.Layer):
    def __init__(self, num_classes=2, stages=None):
        super(BiSeNet, self).__init__()
        if stages is None:
            self.stages = [4, 4, 4]
        else:
            self.stages = stages
        self.num_chasses = num_classes

        self.convX1 = Conv2D(3, 32, 3, 2, 1)
        self.convX2 = Conv2D(32, 64, 3, 2, 1)

        for n1 in range(self.stages[0]):
            if n1 == 0:
                self.stdcX3s = paddle.nn.LayerList([STDC(64, 256, 2)])
                self.stdcX4s = paddle.nn.LayerList([STDC(256, 512, 2)])
                self.stdcX5s = paddle.nn.LayerList([STDC(512, 1024, 2)])
            else:
                self.stdcX3s.append(STDC(256, 256))
                self.stdcX4s.append(STDC(512, 512))
                self.stdcX5s.append(STDC(1024, 1024))

        self.convX6 = Conv2D(1024, 1024, 1, 1, 0)

        self.contextpath = ContextPath()
        self.ffm = FFM()
        self.conv_out = BisNetOutput(256, 256, num_classes)

        self.avg_pool = nn.AvgPool2D(3, 2, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2D(1)
        self.linear_1 = nn.Linear(1024, 1024)
        self.linear_2 = nn.Linear(1024, 1000)

    def forward(self, inputs):
        # Encoder
        # stage1 ~ stage2
        H, W = inputs.shape[2:]
        feat_res2 = self.convX1(inputs)
        feat_res4 = self.convX2(feat_res2)

        # stage3
        for idx, s3 in enumerate(self.stdcX3s):
            feat_res8 = s3(feat_res4) if idx == 0 else s3(feat_res8)
        # stage4
        for idx, s4 in enumerate(self.stdcX4s):
            feat_res16 = s4(feat_res8) if idx == 0 else s4(feat_res16)
        # stage5
        for idx, s5 in enumerate(self.stdcX5s):
            feat_res32 = s5(feat_res16) if idx == 0 else s5(feat_res32)

        # stage6
        stage6 = self.convX6(feat_res32)
        stage6 = self.global_avg_pool(stage6).flatten(start_axis=1)
        stage6 = self.linear_1(stage6)
        stage6 = self.linear_2(stage6)

        # Decoder
        feat_cp8, feat_cp16 = self.contextpath(feat_res8, feat_res16, feat_res32)
        feat_fuse = self.ffm(feat_res8, feat_cp8)
        feat_out = self.conv_out(feat_fuse)
        feat_out = F.upsample(feat_out, size=(H, W), mode="bilinear")
        return stage6, feat_out
