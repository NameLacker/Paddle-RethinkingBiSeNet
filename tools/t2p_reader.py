# -*- coding: utf-8 -*-
# ===========================================
# @Time    : 2021/8/16 上午9:59
# @Author  : shutao
# @FileName: t2p_reader.py
# @remark  : 
# 
# @Software: PyCharm
# Github 　： https://github.com/NameLacker
# ===========================================

import os.path as osp
import numpy as np
import json
import os

from paddle.vision import transforms
from paddle.io import Dataset

from .transforms import *


class CityScapes(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train',
                 randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5)):
        super(CityScapes, self).__init__()
        assert mode in ('train', 'val', 'test', 'trainval')
        self.rootpth = rootpth
        self.datalist_file = osp.join(rootpth, mode + '.list')
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255

        with open('./cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        self.datalist = []
        self.create_datalist()

        # pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomScale(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            RandomCrop(cropsize)
        ])

    def create_datalist(self):
        with open(self.datalist_file, 'r') as f:
            filelist = f.readlines()
        for file in filelist:
            data = {"image_path": None, "gt_path": None}
            image_path, gt_path = file.replace('\n', '').split(' ')
            data["image_path"] = os.path.join(self.rootpth, image_path)
            data["gt_path"] = os.path.join(self.rootpth, gt_path)
            self.datalist.append(data)

    def __getitem__(self, item):
        fn = self.datalist[item]
        impth = fn["image_path"]
        lbpth = fn["gt_path"]
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)
        if self.mode == 'train' or self.mode == 'val':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = self.convert_labels(label)
        return img, label

    def __len__(self):
        return len(self.datalist)

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label
