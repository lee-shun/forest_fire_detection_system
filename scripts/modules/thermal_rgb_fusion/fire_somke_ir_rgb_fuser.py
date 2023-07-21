#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: fire_somke_ir_rgb_fuser.py
#
#   @Author: Shun Li
#
#   @Date: 2021-11-28
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
# ------------------------------------------------------------------------------

import threading
from cv_bridge import CvBridge, CvBridgeError
from heat_ir_threshold_locater import HeatIrThresholdLocater
from fire_smoke_rgb_resnet_classifier import FireSmokeRgbResnetClassifier
from sensor_msgs.msg import Image
import message_filters
from forest_fire_detection_system.msg import SingleFireFuse
import rospy
from tools.custom_logger import Log
import os
import sys

PKG_PATH = os.path.expanduser('~/catkin_ws/src/forest_fire_detection_system/')
sys.path.append(PKG_PATH + 'scripts/')


class ThreadWithReturn(threading.Thread):
    def __init__(self, func, args=()):
        super(ThreadWithReturn, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result


class FireSmokeIrRgbFuser(object):
    def __init__(self):
        self.log = Log(self.__class__.__name__).getlog()
        self.ir_locater = HeatIrThresholdLocater()
        self.rgb_classifier = FireSmokeRgbResnetClassifier()

        # ros
        rospy.wait_for_message(
            "forest_fire_detection_system/main_camera_ir_image", Image)
        self.ir_sub = message_filters.Subscriber(
            'forest_fire_detection_system/main_camera_ir_image', Image)

        rospy.wait_for_message(
            "forest_fire_detection_system/main_camera_rgb_image", Image)
        self.rgb_sub = message_filters.Subscriber(
            'forest_fire_detection_system/main_camera_rgb_image', Image)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.ir_sub, self.rgb_sub],
            queue_size=1,
            slop=0.1,
            allow_headerless=True)
        self.ts.registerCallback(self.ir_rgb_callback)

        self.fuse = SingleFireFuse()
        self.fuse_pub = rospy.Publisher(
            "forest_fire_detection_system/single_fire_in_fuse_images",
            SingleFireFuse,
            queue_size=10)

        # images
        self.convertor = CvBridge()
        self.cv_ir_img = None
        self.cv_rgb_img = None

        rospy.sleep(2.0)
        self.log.info("initialize done! %s", self.__class__.__name__)

    def ir_rgb_callback(self, ir, rgb):
        try:
            self.cv_ir_image = self.convertor.imgmsg_to_cv2(ir, 'bgr8')
        except CvBridgeError as err:
            self.log.warning(err)

        try:
            self.cv_rgb_image = self.convertor.imgmsg_to_cv2(rgb, 'bgr8')
        except CvBridgeError as err:
            self.log.warning(err)

    def ir_locate(self, ir_image):
        ir_heat_pos_x, ir_heat_pos_y = self.ir_locater.locate(ir_image)
        return ir_heat_pos_x, ir_heat_pos_y

    def rgb_classify(self, rgb_image):
        pred_class = self.rgb_classifier.classify(rgb_image)
        return pred_class

    def run(self):
        while not rospy.is_shutdown():
            ir_thread = ThreadWithReturn(FireSmokeIrRgbFuser.ir_locate,
                                         args=(self, self.cv_ir_image))
            rgb_thread = ThreadWithReturn(FireSmokeIrRgbFuser.rgb_classify,
                                          args=(self, self.cv_rgb_image))

            ir_thread.start()
            rgb_thread.start()
            ir_thread.join()
            rgb_thread.join()

            ir_heat_pos_x, ir_heat_pos_y = ir_thread.get_result()
            pred_class = rgb_thread.get_result()

            heat_find = ir_heat_pos_y != -1 and ir_heat_pos_x != -1
            smoke_fire_find = pred_class == 'fire' or pred_class == 'smoke'

            self.fuse.header.stamp = rospy.Time().now()
            self.fuse.header.frame_id = "H20T_FUSE"

            if heat_find or smoke_fire_find:
                self.log.info("heat or smoke_fire detected!")
                self.fuse.ir_img_x = ir_heat_pos_x
                self.fuse.ir_img_y = ir_heat_pos_y

            else:
                self.log.info("not potenttial single fire pos detected!")
                self.fuse.ir_img_x = -1
                self.fuse.ir_img_y = -1

            rospy.Rate(5).sleep()


if __name__ == '__main__':
    rospy.init_node("fire_somke_ir_rgb_fuser_node", anonymous=True)
    fuser = FireSmokeIrRgbFuser()
    fuser.run()
