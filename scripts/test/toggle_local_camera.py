#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: toggle_loacl_camera.py
#
#   @Author: Shun Li
#
#   @Date: 2021-09-24
#
#   @Email: 2015097272@qq.com
#
#   @Description: read the local camera with opencv and the convert to ros img and pub
#
# ------------------------------------------------------------------------------

import os
import sys

PKG_PATH = os.path.expanduser('~/catkin_ws/src/forest_fire_detection_system/')
sys.path.append(PKG_PATH + 'scripts/')

from tools.custom_logger import Log

import cv2
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy


def cv_bridge_converter(cv_img):

    bridge = CvBridge()
    ros_frame = bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
    return ros_frame


def hand_converter(cv_img):

    ros_frame = Image()
    header = Header(stamp=rospy.Time.now())
    header.frame_id = "Camera"
    ros_frame.header = header
    ros_frame.height = cv_img.shape[0]
    ros_frame.width = cv_img.shape[1]
    ros_frame.encoding = "bgr8"
    # ros_frame.step = 1920
    ros_frame.data = np.array(cv_img).tobytes()

    return ros_frame


if __name__ == "__main__":
    log = Log(__name__).getlog()

    capture = cv2.VideoCapture(0)

    rospy.init_node('Camera', anonymous=True)
    rate = rospy.Rate(10)
    image_pub = rospy.Publisher("dji_osdk_ros/main_camera_images",
                                Image,
                                queue_size=10)

    while not rospy.is_shutdown():
        ret, frame = capture.read()

        if frame is not None:
            ros_frame = hand_converter(frame)
            image_pub.publish(ros_frame)
            log.info("publishing camera!")
        else:
            log.warning("None frame! waiting for the frame!")
            continue

        rate.sleep()

    capture.release()
    cv2.destroyAllWindows()
    log.info("Done!")
