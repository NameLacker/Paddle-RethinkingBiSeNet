# -*- coding: utf-8 -*-
"""
========================================
@Time   ：2021/8/6 17:16
@Auther ：shutao
@File   ：modules.py
@IDE    ：PyCharm
@Github ：https://github.com/NameLacker
@Gitee  ：https://gitee.com/nameLacker
========================================
"""

import paddle

from paddle import nn
from paddle.nn import functional as F


class Conv2D(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0 if True else []):
        """
        卷积 --> 归一化 --> Relu激活
        :param in_channels:　输入层数
        :param out_channels:　输出层数
        :param kernel_size:　卷积核尺寸
        :param stride:　步长
        :param padding:　补0外衬
        """
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2D(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.bn = nn.BatchNorm(self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播
        :param x: 输入
        :return: 输出
        """
        net = self.conv(x)
        net = self.bn(net)
        net = self.relu(net)
        return net


class STDC(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super(STDC, self).__init__()
        self.stride = stride

        self.conv1 = Conv2D(in_channels, out_channels // 2, 1, 1, 0)
        if stride == 1:
            self.conv2 = Conv2D(out_channels // 2, out_channels // 4, 3, 1, 1)
        elif stride == 2:
            self.conv2 = Conv2D(out_channels // 2, out_channels // 4, 3, 2, 1)
        else:
            raise ValueError("请选择正确的步长参数，当前stride为： {}".format(stride))
        self.conv3 = Conv2D(out_channels // 4, out_channels // 8, 3, 1, 1)
        self.conv4 = Conv2D(out_channels // 8, out_channels // 8, 3, 1, 1)
        self.avg_pool = nn.AvgPool2D(3, 2, 1)

    def forward(self, ftp):
        fp_1 = self.conv1(ftp)
        fp_2 = self.conv2(fp_1)
        fp_3 = self.conv3(fp_2)
        fp_4 = self.conv4(fp_3)
        if self.stride == 2:
            fp_1 = self.avg_pool(fp_1)

        return paddle.concat([fp_1, fp_2, fp_3, fp_4], axis=1)


class ARM(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(ARM, self).__init__()
        self.conv = Conv2D(in_channels, out_channels, 3, 1, 1)
        self.conv_atten = nn.Conv2D(out_channels, out_channels, 1)
        self.bn_atten = nn.BatchNorm2D(out_channels)
        self.sigmoid_atten = nn.Sigmoid()
        self.global_pool = nn.AdaptiveAvgPool2D(1)

    def forward(self, ftp):
        feat = self.conv(ftp)
        atten = self.conv_atten(feat)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        atten = self.global_pool(atten)

        return paddle.multiply(feat, atten)


class FFM(nn.Layer):
    def __init__(self, out_channels=256):
        super(FFM, self).__init__()
        self.out_channels = out_channels

        self.convblk = None
        self.global_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(out_channels, out_channels // 4, 1)
        self.conv2 = nn.Conv2D(out_channels // 4, out_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = paddle.concat([fsp, fcp], axis=1)
        # 重新定义算子
        self.convblk = Conv2D(fcat.shape[1], self.out_channels, 1)

        feat = self.convblk(fcat)
        atten = self.global_pool(feat)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)

        feat_atten = paddle.multiply(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class Head(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2D(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, ftp):
        feat = self.conv1(ftp)
        feat = self.conv2(feat)
        feat = self.bn(feat)
        feat = self.relu(feat)

        return feat


class ContextPath(nn.Layer):
    def __init__(self):
        super(ContextPath, self).__init__()

        self.arm16 = ARM(512, 128)
        self.arm32 = ARM(1024, 128)

        self.global_avgpool = nn.AdaptiveAvgPool2D(1)
        self.conv_avg = Conv2D(1024, 128, 1)

        self.conv_head32 = Conv2D(128, 128, 3, 1, 1)
        self.conv_head16 = Conv2D(128, 128, 3, 1, 1)

    def forward(self, feat8, feat16, feat32):
        H8, W8 = feat8.shape[2:]
        H16, W16 = feat16.shape[2:]
        H32, W32 = feat32.shape[2:]

        avg = self.global_avgpool(feat32)
        avg = self.conv_avg(avg)
        avg_up = F.upsample(avg, size=(H32, W32))

        feat32_arm = self.arm32(feat32)
        feat32_arm = feat32_arm + avg_up
        feat32_up = F.upsample(feat32_arm, size=(H16, W16))
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_arm = feat16_arm + feat32_up
        feat16_up = F.upsample(feat16_arm, size=(H8, W8))
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up


class BisNetOutput(nn.Layer):
    def __init__(self, in_channels, mid_channels, n_classes):
        super(BisNetOutput, self).__init__()
        self.conv = Conv2D(in_channels, mid_channels, 3, 1, 1)
        self.conv_out = nn.Conv2D(mid_channels, n_classes, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x
