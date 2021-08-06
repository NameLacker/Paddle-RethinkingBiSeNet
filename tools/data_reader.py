# -*- coding: utf-8 -*-
"""
========================================
@Time   ：2021/8/6 17:17
@Auther ：shutao
@File   ：data_reader.py
@IDE    ：PyCharm
@Github ：https://github.com/NameLacker
@Gitee  ：https://gitee.com/nameLacker
========================================
"""

import numpy as np
import cv2 as cv

from paddle.io import Dataset

from tools.utils import get_configuration

dataset_cfg = get_configuration()["dataset_config"]


class SegDataset(Dataset):
    def __init__(self, is_test=False):
        super(SegDataset, self).__init__()
        self.img_size = dataset_cfg["img_size"]

    def __getitem__(self, item):
        img = np.random.random((3, 400, 400))
        label = np.random.random((2, 400, 400))
        return img, label

    def __len__(self):
        return 10
