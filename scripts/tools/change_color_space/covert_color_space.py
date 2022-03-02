#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: covert_color_space.py
#
#   @Author: Shun Li
#
#   @Date: 2022-01-10
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
# ------------------------------------------------------------------------------

import cv2
import numpy as np

image_name = "1"
# image_name = "2"
# image_name = "3" # NOTE: difference
# image_name = "4"
ir_img = cv2.imread(image_name+".jpg")
b, g, r = cv2.split(ir_img)
cv2.imwrite(image_name + "_b.jpg", b)
cv2.imwrite(image_name + "_g.jpg", g)
cv2.imwrite(image_name + "_r.jpg", r)

hsv_ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_ir_img)
cv2.imwrite(image_name + "_h.jpg", h)
cv2.imwrite(image_name + "_s.jpg", s)
cv2.imwrite(image_name + "_v.jpg", v)

_, binary = cv2.threshold(h, 80, 255, cv2.THRESH_BINARY_INV)
cv2.imshow(image_name + "_binary", binary)
cv2.waitKey(0)
cv2.imwrite(image_name + "_h_threshold.jpg", binary)

kernel = np.ones((2, 2), dtype="uint8")
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imshow(image_name + "_opening", opening)
cv2.waitKey(0)
cv2.imwrite(image_name + "_h_opening.jpg", opening)
