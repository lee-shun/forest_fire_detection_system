#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: video_to_images.py
#
#   @Author: Shun Li
#
#   @Date: 2021-10-14
#
#   @Email: 2015097272@qq.com
#
#   @Description: 
#
#------------------------------------------------------------------------------


import os
import cv2

SKIP_FRAME = 100
VIDEO_NAME = "../DJI_0026.MOV"

if __name__ == "__main__":

    os.system("rm -rf output_images/")
    os.system("mkdir output_images")

    capture = cv2.VideoCapture(VIDEO_NAME)
    all_frame_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    print("start converting", VIDEO_NAME)
    frame_num = 0
    img_num = 0

    while(frame_num<all_frame_num):

        ret, frame = capture.read()
        frame_num = frame_num + 1

        if frame_num % SKIP_FRAME == 0 :
            # Key frame :)
            cv2.imwrite("output_images/img"+str(img_num)+".jpg", frame)
            img_num = img_num + 1

        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    capture.release()
    print("Quit!")
