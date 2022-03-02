#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVLab. All rights reserved.
#
#   @Filename: print_img_boundary_in_aligned_img.py
#
#   @Author: Shun Li
#
#   @Date: 2021-09-20
#
#   @Email: 2015097272@qq.com
#
#   @Description: some functions between the opencv and pytorch
#
# ------------------------------------------------------------------------------

import cv2
from cv_bridge import CvBridge
import numpy as np
import rospy
from sensor_msgs.msg import Image

class ImgBoundaryFinder(object):
    def __init__(self):
        self.ros_image = Image()
        self.convertor = CvBridge()

        rospy.wait_for_message("dji_osdk_ros/main_camera_images", Image)
        self.image_sub = rospy.Subscriber("dji_osdk_ros/main_camera_images",
                                          Image, self.image_cb)
    def image_cb(self, msg):
        self.ros_image = msg
        full_img = self.convertor.imgmsg_to_cv2(self.ros_image, 'bgr8')

        img_gray = cv2.cvtColor(full_img,cv2.COLOR_RGB2GRAY)

        img_bin = img_gray>0

        img_white_arg = np.argwhere(img_bin==1)
        print(img_white_arg)

        cv2.imshow("bin", full_img[336:1106, :,:])
        cv2.waitKey(3)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node("find_img_boundary_node", anonymous=True)
    detector = ImgBoundaryFinder()
    detector.run()
