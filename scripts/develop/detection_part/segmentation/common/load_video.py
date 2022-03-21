#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: load_video.py
#
#   @Author: Shun Li
#
#   @Date: 2021-10-12
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
# ------------------------------------------------------------------------------

from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
import torch
from torch2trt import TRTModule
from model import UNET

import sys
import time

sys.path.append('../../')
from tools.tensor2cv2 import tensor_to_cv, draw_mask


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

detector_trt = UNET(in_channels=3, out_channels=1).to(DEVICE)
detector_trt.load_state_dict(torch.load("./ModelParams/final.pth"))
detector_trt.eval()

# detector_trt = TRTModule().to(DEVICE)
# detector_trt.load_state_dict(torch.load("./ModelParams/final_trt.pth"))
# print("loading params from: final_trt.pth")

# capture = cv2.VideoCapture(
#     "/home/ls/dataset/chimney_Somke_segmentation/videoplayback.mp4")
capture = cv2.VideoCapture("/media/ls/WORK/FLIGHT_TEST/M300/DJI_202110171037_002/DJI_20211017111617_0004_Z.MP4")
# capture = cv2.VideoCapture(
#     "/home/ls/dataset/NAVlab_smoke_database/Royal_Mountain_park-1/Jingling_Royal_Mountain_park.MOV")
# capture = cv2.VideoCapture( "/media/ls/WORK/FIRE_VIDEO/smoke_cake_effect.MP4")

val_transforms = A.Compose([
    A.Resize(height=255, width=255),
    A.Normalize(),
    ToTensorV2(),
], )

while (1):
    start_time = time.time()

    ret, frame = capture.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    augmentations = val_transforms(image=img_rgb)
    img_ = augmentations['image']
    img_ = img_.float().unsqueeze(0).to(device=DEVICE)

    with torch.no_grad():
        preds = torch.sigmoid(detector_trt(img_))
        preds = (preds > 0.5)

    cv_mask = tensor_to_cv(preds[0].cpu())

    masked_img = draw_mask(cv2.resize(frame, (255, 255)), cv_mask)

    cv_3_mask = cv2.merge((cv_mask, cv_mask, cv_mask))

    show_img = cv2.hconcat([masked_img, cv_3_mask])
    cv2.imshow("mask", show_img)

    end_time = time.time()
    time_dura = end_time - start_time
    print("FPS:%.2f" % (1 / time_dura))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
print("end")
