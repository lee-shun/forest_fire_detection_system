#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: load_image.py
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

import os
import sys

PKG_PATH = os.path.expanduser('~/catkin_ws/src/forest_fire_detection_system/')
sys.path.append(PKG_PATH + 'scripts/')

from tools.tensor2cv2 import tensor_to_cv, draw_mask

import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from model import UNET

from torch2trt import torch2trt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

val_transforms = A.Compose([
    A.Resize(height=480, width=720),
    A.Normalize(),
    ToTensorV2(),
], )

img_rgb = np.array(
    Image.open(
        "/home/ls/dataset/Unet_Smoke_segmentation/flight_test_images/DJI_20211017111122_0003_Z_MP4_5.png"
    ))

img_cv = cv2.imread(
    "/home/ls/dataset/Unet_Smoke_segmentation/flight_test_images/DJI_20211017111122_0003_Z_MP4_5.png"
)

augmentations = val_transforms(image=img_rgb)
img_ = augmentations['image']
img_ = img_.float().unsqueeze(0).to(DEVICE)

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load("./ModelParams/final.pth"))
model.eval()

with torch.no_grad():
    preds = torch.sigmoid(model(img_))
    preds = (preds > 0.50)

# original model
cv_mask = tensor_to_cv(preds[0].cpu())
cv2.imshow("cv", cv_mask)
print(cv_mask[200, 125])
cv2.waitKey(0)

masked_img = draw_mask(cv2.resize(img_cv, (720, 480)), cv_mask)
cv2.imshow("cv", masked_img)
cv2.waitKey(0)

# optimied model with thesorrt
init_x = torch.ones((1, 3, 255, 255)).cuda()
detector_trt = torch2trt(model, [init_x], fp16_mode=False)

with torch.no_grad():
    preds = torch.sigmoid(detector_trt(img_))
    preds = (preds > 0.5)

cv_mask = tensor_to_cv(preds[0].cpu())
cv2.imshow("cv", cv_mask)
cv2.waitKey(0)

masked_img = draw_mask(cv2.resize(img_cv, (255, 255)), cv_mask)
cv2.imshow("cv", masked_img)
cv2.waitKey(0)

# torch.save(detector_trt.state_dict(), 'final_trt.pth')
