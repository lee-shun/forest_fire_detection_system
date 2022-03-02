#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: pub_local_video.py
#
#   @Author: Shun Li
#
#   @Date: 2021-10-13
#
#   @Email: 2015097272@qq.com
#
#   @Description: Convert the local video to the to the topic that
#   detection_firedetection_fire_smoke_node needs.
#
# ------------------------------------------------------------------------------

import os
import sys

PKG_PATH = os.path.expanduser('~/catkin_ws/src/forest_fire_detection_system/')
sys.path.append(PKG_PATH + 'scripts/')

import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image

from tools.custom_logger import Log

RESZIE_WIDTH = 720
RESZIE_HEIGHT = 480

if __name__ == "__main__":

    rospy.init_node('load_local_video_node', anonymous=True)
    log = Log(__name__).getlog()

    # video_name = os.path.expanduser(
    #         "/media/ls/WORK/FLIGHT_TEST/M300/DJI_202110171037_002/DJI_20211017104441_0001_Z.MP4")
    video_name = os.path.expanduser(
            "/media/ls/WORK/FLIGHT_TEST/M300/DJI_202110171037_002/DJI_20211017111617_0004_W.MP4")
    # video_name = os.path.expanduser("/media/ls/WORK/FIRE_VIDEO/smoke_cake_effect.MP4")
    # video_name = os.path.expanduser(
    #     "~/dataset/NAVlab_smoke_database/Royal_Mountain_park-1/Jingling_Royal_Mountain_park.MOV"
    # )
    # video_name = os.path.expanduser("~/dataset/chimney_Somke_segmentation/videoplayback.mp4")
    capture = cv2.VideoCapture(video_name)
    bridge = CvBridge()

    log.info("video from: " + video_name)

    rate = rospy.Rate(20)
    # image_pub = rospy.Publisher(
    #     "dji_osdk_ros/main_camera_images", Image, queue_size=1)
    image_pub = rospy.Publisher(
        "forest_fire_detection_system/main_camera_rgb_image",
        Image,
        queue_size=1)

    while not rospy.is_shutdown():
        ret, frame = capture.read()

        if frame is not None:
            # cv2.imshow("frame", frame)
            # cv2.waitKey(3)
            frame = cv2.resize(frame, (RESZIE_WIDTH, RESZIE_HEIGHT))
            ros_img = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            image_pub.publish(ros_img)
        else:
            # cv2.destroyAllWindows()
            log.warning("None frame! end of video")

        rate.sleep()

    capture.release()
