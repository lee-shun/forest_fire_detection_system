#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: tensor2cv2.py
#
#   @Author: Shun Li
#
#   @Date: 2021-12-03
#
#   @Email: 2015097272@qq.com
#
#   @Description: 
#
#------------------------------------------------------------------------------



import numpy as np
import cv2
import torch


def cv_to_tesnor(cv_img, re_width, re_height, device):
    """

    Description: This function convert "BGR" cv image[H, W, C] --> tensor(1, C, H, W)
                and "C" usually should be 3

    Note: Usually, we don not need this function, cause the albumentations can
            convert the numpy to tensor with ToTensorV2().
            Or, the torchvision can also do that too~

    """

    # cv(BGR) --> tensor(RGB)
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    # resize the image for tensor
    img = cv2.resize(img, (re_width, re_height))

    # change the shape order and add bathsize
    img_ = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)

    return img_.to(device)


def tensor_to_cv(ten_img_cpu):
    """

    Note: The tensor[B, C, H, W] could be any value, but cv_image[H, W, C] should be in (0~255)
    <uint8>

    """

    # tensor --> numpy
    np_array = ten_img_cpu.detach().numpy()

    # normalize
    maxValue = np_array.max()
    np_array = (np_array / maxValue) * 255
    mat = np.uint8(np_array)

    # change the dimension shape to fit cv image
    mat = np.transpose(mat, (1, 2, 0))

    return mat


def draw_mask(cv_org_img, cv_mask):

    signle_mask = cv_mask[:, :, 0]
    target_channel = cv_org_img[:, :, 2]

    index = (signle_mask > 254)
    target_channel[index] = (signle_mask[index] * 0.7 +
                             target_channel[index] * 0.3).astype(np.uint8)

    cv_org_img[:, :, 2] = target_channel

    return cv_org_img
