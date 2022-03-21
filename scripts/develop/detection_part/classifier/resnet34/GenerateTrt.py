#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: GenerateTrt.py
#
#   @Author: Shun Li
#
#   @Date: 2021-09-29
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
# ------------------------------------------------------------------------------

import torch
from resnet34_model import Resnet34
from torch2trt import torch2trt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Resnet34(img_channels=3, num_classes=3).to(DEVICE)
checkpoint = torch.load('./C_processingparam_resnet34.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


init_x = torch.ones((1, 3, 255, 255)).cuda()
detector_trt = torch2trt(model, [init_x])


torch.save(detector_trt.state_dict(), 'resnet34_trt.pth')
