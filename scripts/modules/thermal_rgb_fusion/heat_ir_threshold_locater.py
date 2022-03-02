#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: heat_ir_threshold_locater.py
#
#   @Author: Shun Li
#
#   @Date: 2021-11-28
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
#------------------------------------------------------------------------------

import os
import sys

PKG_PATH = os.path.expanduser('~/catkin_ws/src/forest_fire_detection_system/')
sys.path.append(PKG_PATH + 'scripts/')

from tools.custom_logger import Log

import cv2
from cv_bridge import CvBridge, CvBridgeError
from forest_fire_detection_system.msg import SingleFireIR
import numpy as np
import rospy
from sensor_msgs.msg import Image
import yaml
from yaml import CLoader


class HeatIrThresholdLocater():
    def __init__(self):
        self.log = Log(self.__class__.__name__).getlog()

        config = open(PKG_PATH + "config/H20T_Camera.yaml")
        self.H20T = yaml.load(config, Loader=CLoader)

        self.ir_img = np.zeros(
            (self.H20T["pure_IR_height"], self.H20T["pure_IR_width"], 3),
            dtype='uint8')
        self.ros_image = Image()

        self.convertor = CvBridge()
        self.single_fire_ir = SingleFireIR()
        self.single_fire_ir.target_type = self.single_fire_ir.IS_UNKNOWN
        self.log.info("initialize done! %s", self.__class__.__name__)

    def image_cb(self, msg):
        self.ros_image = msg
        try:
            self.ir_img = self.convertor.imgmsg_to_cv2(self.ros_image, 'bgr8')
        except CvBridgeError as err:
            self.log.warning(err)

    def sliding_window(self, image, stepSize=10, windowSize=[20, 20]):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    def locate(self, ir_img, windowSize=[40, 40], stepSize=20):
        _, binary = cv2.threshold(ir_img[:, :, 2], 25, 255, cv2.THRESH_BINARY)

        # opening operation
        kernel = np.ones((2, 2), dtype="uint8")
        opening = cv2.morphologyEx(binary,
                                   cv2.MORPH_OPEN,
                                   kernel,
                                   iterations=2)
        judge_list = []
        coord_list = []
        img_x = -1
        img_y = -1
        for (x, y, patch) in self.sliding_window(opening, stepSize,
                                                 windowSize):
            coord_list.append([x, y])
            judje = patch > 0
            judge_list.append(np.count_nonzero(judje))

        if (np.count_nonzero(judge_list) != 0):

            best_index = judge_list.index(max(judge_list))
            best_pos = coord_list[best_index]

            img_x = best_pos[0] + windowSize[0] / 2
            img_y = best_pos[1] + windowSize[1] / 2
            self.log.debug("heat point x,y: (%d,%d)", img_x, img_y)

        else:
            self.log.info("no potential fire currently!")

        return img_x, img_y

    def run(self):
        rospy.wait_for_message(
            "forest_fire_detection_system/main_camera_ir_image", Image)
        self.image_sub = rospy.Subscriber(
            "forest_fire_detection_system/main_camera_ir_image",
            Image,
            self.image_cb,
            queue_size=1)
        self.heat_point_pub = rospy.Publisher(
            "forest_fire_detection_system/single_fire_in_ir_image",
            SingleFireIR,
            queue_size=10)
        rospy.sleep(2.0)
        self.log.info("sub and pud registing done...")

        # video saver
        res_video_saver = cv2.VideoWriter('threshold_locater_result.avi',
                                          cv2.VideoWriter_fourcc(*'DIVX'), 5,
                                          (720, 480))

        windowSize = [40, 40]
        while not rospy.is_shutdown():
            # STEP: 1 locate the heat center pos
            img_x, img_y = self.locate(self.ir_img, windowSize)

            # STEP: 2 draw the result, save
            if img_x == -1 or img_y == -1:
                self.single_fire_ir.target_type = self.single_fire_ir.IS_BACKGROUND
            else:
                self.single_fire_ir.target_type = self.single_fire_ir.IS_HEAT
                retan_x = int(img_x - windowSize[0] / 2)
                retan_y = int(img_y - windowSize[1] / 2)
                cv2.rectangle(self.ir_img, (retan_x, retan_y),
                            (retan_x + windowSize[0], retan_y + windowSize[1]),
                            (0, 255, 0), 2)

            resized = cv2.resize(self.ir_img, (720, 480))
            res_video_saver.write(resized)

            # STEP: 3 publish
            self.single_fire_ir.header.stamp = rospy.Time.now()
            self.single_fire_ir.header.frame_id = "H20T_IR"
            self.single_fire_ir.img_x = img_x
            self.single_fire_ir.img_y = img_y
            self.single_fire_ir.img_width = self.ir_img.shape[1]
            self.single_fire_ir.img_height = self.ir_img.shape[0]
            self.heat_point_pub.publish(self.single_fire_ir)

            rospy.Rate(15).sleep()

        res_video_saver.release()
        self.log.info("end of saveing result!")


if __name__ == '__main__':
    rospy.init_node("heat_ir_threshold_locater_node", anonymous=True)
    detector = HeatIrThresholdLocater()
    detector.run()
