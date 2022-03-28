#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: load_save_local_video.py
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

import cv2

SAVED_VIDEO_SIZE = (720, 480)
VIDEO_NAME = "../DJI_0026.MOV"
video_saver = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 5,
                              SAVED_VIDEO_SIZE)
if __name__ == "__main__":

    capture = cv2.VideoCapture(VIDEO_NAME)
    all_frame_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(all_frame_num):
        ret, frame = capture.read()
        print("read the frame %d\n", i)
        resized = cv2.resize(frame, SAVED_VIDEO_SIZE)
        video_saver.write(resized)

    video_saver.release()
    capture.release()
    print("Quit!")
