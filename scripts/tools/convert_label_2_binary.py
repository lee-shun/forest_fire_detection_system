#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: convert_label_2_binary.py
#
#   @Author: Shun Li
#
#   @Date: 2021-10-14
#
#   @Email: 2015097272@qq.com
#
#   @Description: convert the original binary to balck-white{0,255} binary
#
#------------------------------------------------------------------------------

import os
import cv2

IMAGE_PATH = "label_images/"

def make_binary(img, threshold=0):
    # ret, _ = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    # return ret
    img[img>threshold]=255
    return img

if __name__ == "__main__":
    image_names = os.listdir(IMAGE_PATH)
    print(image_names)

    for name in image_names:
        img = cv2.imread(IMAGE_PATH+name,0)
        binary_img = make_binary(img)

        cv2.imwrite("./binary_label/"+name, binary_img)
