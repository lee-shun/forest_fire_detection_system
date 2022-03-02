#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: seperate_original_image.py
#
#   @Author: Shun Li
#
#   @Date: 2021-11-07
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
#------------------------------------------------------------------------------

import os
from cv_bridge import CvBridge
import numpy as np
import rospy
from sensor_msgs.msg import Image
import yaml
from yaml import CLoader

PKG_PATH = os.path.expanduser('~/catkin_ws/src/forest_fire_detection_system')


class OriginalImageSeperator(object):
    def __init__(self):
        # read the camera parameters
        config_path = open(PKG_PATH + "/config/H20T_Camera.yaml")
        self.H20T = yaml.load(config_path, Loader=CLoader)

        self.full_img = np.zeros(
            (self.H20T["full_img_height"], self.H20T["full_img_width"], 3),
            dtype='uint8')
        self.pure_ir_img = np.zeros(
            (self.H20T["pure_IR_height"], self.H20T["pure_IR_width"], 3),
            dtype='uint8')
        self.pure_rgb_img = np.zeros(
            (self.H20T["pure_RGB_height"], self.H20T["pure_RGB_width"], 3),
            dtype='uint8')

        self.ros_image = Image()
        self.convertor = CvBridge()

        rospy.wait_for_message("dji_osdk_ros/main_camera_images", Image)
        self.image_sub = rospy.Subscriber("dji_osdk_ros/main_camera_images",
                                          Image, self.image_cb)
        self.image_ir_pub = rospy.Publisher(
            "forest_fire_detection_system/main_camera_ir_image",
            Image,
            queue_size=10)
        self.image_rgb_pub = rospy.Publisher(
            "forest_fire_detection_system/main_camera_rgb_image",
            Image,
            queue_size=10)

    def image_cb(self, msg):
        self.ros_image = msg
        self.full_img = self.convertor.imgmsg_to_cv2(self.ros_image, 'bgr8')

        # 1920 x 1440
        # rospy.loginfo("ros Image size(W x H): %d x %d", self.ros_image.width,
        #         self.ros_image.height)
        # rospy.loginfo("cv Image size(W x H): %d x %d", full_img.shape[1],
        #         full_img.shape[0])

        self.pure_ir_img = self.full_img[
            self.H20T["upper_bound"]:self.H20T["lower_bound"], :self.
            H20T["pure_IR_width"], :]

        self.pure_rgb_img = self.full_img[
            self.H20T["upper_bound"]:self.H20T["lower_bound"],
            self.H20T["pure_RGB_width"]:, :]

        # print(self.pure_ir_img.shape)
        # cv2.imshow("ir", self.pure_ir_img)
        # cv2.waitKey(1)
        # cv2.imshow("rgb", self.pure_rgb_img)
        # cv2.waitKey(1)

    def run(self):
        while not rospy.is_shutdown():
            ros_ir_img = self.convertor.cv2_to_imgmsg(self.pure_ir_img,
                                                      encoding="bgr8")
            self.image_ir_pub.publish(ros_ir_img)

            ros_rgb_img = self.convertor.cv2_to_imgmsg(self.pure_rgb_img,
                                                       encoding="bgr8")
            self.image_rgb_pub.publish(ros_rgb_img)

            rospy.Rate(10).sleep()


if __name__ == '__main__':
    rospy.init_node("seperate_original_image_node", anonymous=True)
    detector = OriginalImageSeperator()
    detector.run()
