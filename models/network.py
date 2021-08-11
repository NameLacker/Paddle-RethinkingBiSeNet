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
    def __init__(self, num_classes=2, stages=None,
                 use_boundary_2=False, use_boundary_4=False, use_boundary_8=True):
        super(BiSeNet, self).__init__()
        if stages is None:  # STDC2-50
            self.stages = [4, 5, 3]
        else:
            self.stages = stages
        self.num_chasses = num_classes
        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8

        self.convX1, self.convX2, self.stdcX3s, self.stdcX4s, self.stdcX5s = self.create_backbone()

        self.convX6 = Conv2D(1024, 1024, 1, 1, 0)

        self.contextpath = ContextPath()
        self.ffm = FFM()
        self.conv_out = BisNetOutput(256, 256, num_classes)
        self.conv_out16 = BisNetOutput(128, 64, num_classes)
        self.conv_out32 = BisNetOutput(128, 64, num_classes)

        self.conv_out_sp2 = BisNetOutput(32, 64, 1)
        self.conv_out_sp4 = BisNetOutput(64, 64, 1)
        self.conv_out_sp8 = BisNetOutput(256, 64, 1)

        self.avg_pool = nn.AvgPool2D(3, 2, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2D(1)
        self.linear_1 = nn.Linear(1024, 1024)
        self.linear_2 = nn.Linear(1024, 1000)

    def create_backbone(self):
        convX1 = Conv2D(3, 32, 3, 2, 1)
        convX2 = Conv2D(32, 64, 3, 2, 1)

        for n1 in range(self.stages[0]):
            if n1 == 0:
                stdcX3s = paddle.nn.LayerList([STDC(64, 256, 2)])
            else:
                stdcX3s.append(STDC(256, 256))

        for n2 in range(self.stages[1]):
            if n2 == 0:
                stdcX4s = paddle.nn.LayerList([STDC(256, 512, 2)])
            else:
                stdcX4s.append(STDC(512, 512))

        for n3 in range(self.stages[2]):
            if n3 == 0:
                stdcX5s = paddle.nn.LayerList([STDC(512, 1024, 2)])
            else:
                stdcX5s.append(STDC(1024, 1024))
        return convX1, convX2, stdcX3s, stdcX4s, stdcX5s

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

        # stage6 (分类)
        stage6 = self.convX6(feat_res32)
        stage6 = self.global_avg_pool(stage6).flatten(start_axis=1)
        stage6 = self.linear_1(stage6)
        class_preds = self.linear_2(stage6)

        # Decoder
        feat_cp8, feat_cp16 = self.contextpath(feat_res8, feat_res16, feat_res32)

        feat_out_sp2 = self.conv_out_sp2(feat_res2)
        feat_out_sp4 = self.conv_out_sp4(feat_res4)
        feat_out_sp8 = self.conv_out_sp8(feat_res8)

        feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.upsample(feat_out, size=(H, W), mode="bilinear")
        feat_out16 = F.upsample(feat_out16, size=(H, W), mode="bilinear")
        feat_out32 = F.upsample(feat_out32, size=(H, W), mode="bilinear")

        if self.training:  # 训练阶段
            if self.use_boundary_2 and self.use_boundary_4 and self.use_boundary_8:
                return feat_out, feat_out16, feat_out32, feat_out_sp2, feat_out_sp4, feat_out_sp8
            elif (not self.use_boundary_2) and self.use_boundary_4 and self.use_boundary_8:
                return feat_out, feat_out16, feat_out32, feat_out_sp4, feat_out_sp8
            elif (not self.use_boundary_2) and (not self.use_boundary_4) and self.use_boundary_8:
                return feat_out, feat_out16, feat_out32, feat_out_sp8
            else:
                return feat_out, feat_out16, feat_out32
        else:  # 预测阶段
            return feat_out
