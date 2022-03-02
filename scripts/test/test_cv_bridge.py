#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: test_cv_bridge.py
#
#   @Author: Shun Li
#
#   @Date: 2021-09-29
#
#   @Email: 2015097272@qq.com
#
#   @Description: A cv_bridge example from the ros website
#
# ------------------------------------------------------------------------------

from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
import rospy
import sys

class image_converter:

    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic_2", Image, queue_size=10)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            '/camera/rgb/image_raw', Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        (rows, cols, channels) = cv_image.shape
        if cols > 60 and rows > 60:
            cv2.circle(cv_image, (50, 50), 10, 255)

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)


def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
