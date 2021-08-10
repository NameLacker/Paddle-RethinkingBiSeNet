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
import json
import os

from paddle.io import Dataset
from paddle.vision.transforms import ColorJitter, RandomHorizontalFlip, RandomCrop, Compose, Normalize

from tools.utils import get_configuration

dataset_cfg = get_configuration()["dataset_config"]


class SegDataset(Dataset):
    def __init__(self, is_test=False, datalist_file=None, cropsize=(640, 480)):
        super(SegDataset, self).__init__()
        self.img_h, self.img_w = dataset_cfg["img_size"]
        self.root = dataset_cfg["root"]
        self.is_test = is_test
        self.datalist_file = dataset_cfg["valset_file"] if is_test else dataset_cfg["trainset_file"]
        if datalist_file is not None:  # 读取指定数据列表文件内容
            self.datalist_file = datalist_file
        with open('./cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        self.datalist = []
        self.create_datalist()

        self.trans_train = Compose([RandomHorizontalFlip(), RandomCrop(size=cropsize)])
        self.color_jitter = ColorJitter(0.5, 0.5, 0.5)
        self.normalize = Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC')

    def create_datalist(self):
        with open(self.datalist_file, 'r') as f:
            filelist = f.readlines()
        for file in filelist:
            data = {"image_path": None, "gt_path": None}
            image_path, gt_path = file.replace('\n', '').split(' ')
            data["image_path"] = os.path.join(self.root, image_path)
            data["gt_path"] = os.path.join(self.root, gt_path)
            self.datalist.append(data)

    def covert_label(self, gt_img):
        gt_img = gt_img.astype(np.int64)
        for k, v in self.lb_map.items():
            gt_img[gt_img == k] = v
        gt_img = np.transpose(gt_img, (2, 0, 1))
        return gt_img[0]  # 提取单通道作为标签

    def process(self, img, gt_img):
        # 随机翻转和裁剪(原图和gt图)
        im_lb = dict(im=img, lb=gt_img)
        im_lb = self.trans_train(im_lb)
        img, gt_img = im_lb['im'], im_lb['lb']
        # 随机颜色扰动(原图)
        img = self.color_jitter(img)
        return img, gt_img

    def __getitem__(self, item):
        image_path, gt_path = self.datalist[item]["image_path"], self.datalist[item]["gt_path"]
        img, gt_img = cv.imread(image_path), cv.imread(gt_path)

        if not self.is_test:  # 训练阶段图像增强
            img, gt_img = self.process(img, gt_img)

        # 归一化(原图)
        img = self.normalize(img)
        img = np.transpose(img, (2, 0, 1))
        # 处理gt标签
        gt_img = self.covert_label(gt_img)
        return img, gt_img

    def __len__(self):
        return len(self.datalist)
