# -*- coding: utf-8 -*-
# ===========================================
# @Time    : 2021/8/13 下午3:21
# @Author  : shutao
# @FileName: test.py
# @remark  : 
# 
# @Software: PyCharm
# Github 　： https://github.com/NameLacker
# ===========================================

import cv2 as cv
import numpy as np
import paddle
import json
import os

from paddle.nn import functional as F

from models.network import BiSeNet
from tools.data_reader import CityScapes
from tools.evaluation import evaluate


def read_img(img_path):
    img = cv.imread(img_path)
    cv.imwrite("origin.png", img)
    H, W, C = img.shape

    img = (img - 127.5) / 127.5
    img = np.transpose(img, (2, 0, 1))
    return img, (H, W, C)


def run_test():
    test_dataset = CityScapes(datalist_file="data/val.list", is_test=True)
    test_batch_sampler = paddle.io.DistributedBatchSampler(
        test_dataset, batch_size=4, shuffle=False, drop_last=False)
    test_loader = paddle.io.DataLoader(test_dataset, batch_sampler=test_batch_sampler, num_workers=0, return_list=True)

    params = paddle.load("inferences/model_maxmIOU50_1.1.pdparams")
    net = BiSeNet(num_classes=19, use_boundary_8=True)
    net.load_dict(params)
    net.eval()

    num_eval = test_dataset.__len__()
    hist = np.zeros((19, 19), dtype=np.float32)
    for batch_id, (image, label) in enumerate(test_loader):
        hist += evaluate(image, label, net, n_classes=19)
        print("\r", end="")
        percentage = int(100 * batch_id / num_eval)
        print("Test progress rate: {}%: ".format(percentage), "▋" * (percentage // 2), end="")

    ious = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mIOU50 = np.nanmean(ious)  # 增加对有nan值的处理
    print("测试集mIOU50: {}".format(mIOU50))


def infer(img_path, net, lb_map, idx=0):
    img, img_size = read_img(img_path)
    img_tensor = paddle.to_tensor([img], dtype=paddle.float32)
    logit = net(img_tensor)
    prob = F.softmax(logit, axis=1)
    pred = paddle.argmax(prob, axis=1).numpy()[0]

    res_img = np.zeros(img_size)
    for k, v in lb_map.items():
        res_img[pred == k] = v
    res_img[res_img > 255] = 255
    res_img[res_img < 0] = 0
    cv.imwrite("/home/lacker/work/Dataset/res/{}.png".format(idx), res_img)


def batch_infer(root_path):
    params = paddle.load("inferences/model_maxmIOU50_2.2e+01.pdparams")
    net = BiSeNet(num_classes=19, use_boundary_8=True)
    net.load_dict(params)
    net.eval()

    with open('./cityscapes_info.json', 'r') as fr:
        labels_info = json.load(fr)
    lb_map = {el['trainId']: el['color'] for el in labels_info}

    for idx, _file in enumerate(os.listdir(root_path)):
        if "._" in _file:
            continue
        file_path = os.path.join(root_path, _file)
        print(idx, file_path)
        infer(file_path, net, lb_map, idx)


if __name__ == '__main__':
    batch_infer("/home/lacker/work/Dataset/cityscapes/leftImg8bit/val/frankfurt")
