# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys
import random
import logging
import time
import timeit
from pathlib import Path
import time
import numpy as np

import torch
import torch.nn as nn
from pp_liteseg import PPLiteSeg
import cv2
import torch.nn.functional as F
import datasets



def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--image',
                        help='test image path',
                        default="mainz_000001_009328_leftImg8bit.png",
                        type=str)
    parser.add_argument('--weights',
                        help='cityscape pretrained weights',
                        default="ppliteset_pp2torch_cityscape_pretrained.pth",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb = labelmap_rgb + (labelmap == label)[:, :, np.newaxis] * \
                       np.tile(colors[label],
                               (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def main():
    base_size = 512
    wh = 2
    mean = [0.5, 0.5, 0.5],
    std = [0.5, 0.5, 0.5]
    args = parse_args()

    model = PPLiteSeg()

    model.eval()

    print("ppliteseg:", model)
    ckpt = torch.load(args.weights)
    model = model.cuda()
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)

    img = cv2.imread(args.image)
    imgor = img.copy()
    img = cv2.resize(img, (wh * base_size, base_size))
    image = img.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std

    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)

    #  image = image.permute((2, 0, 1))

    image = image.unsqueeze(0)
    image = image.cuda()
    start = time.time()
    out = model(image)
    end = time.time()
    print("infer time:", end - start, " s")
    out = out[0].squeeze(dim=0)
    outadd = F.softmax(out, dim=0)
    outadd = torch.argmax(outadd, dim=0)
    predadd = outadd.detach().cpu().numpy()
    pred = np.int32(predadd)
    colors = np.random.randint(0, 255, 19 * 3)
    colors = np.reshape(colors, (19, 3))
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)
    pred_color = cv2.resize(pred_color,(imgor.shape[1],imgor.shape[0]))

    im_vis = cv2.addWeighted(imgor, 0.7, pred_color, 0.3, 0)
    cv2.imwrite("results.jpg", im_vis)


if __name__ == '__main__':
    main()
