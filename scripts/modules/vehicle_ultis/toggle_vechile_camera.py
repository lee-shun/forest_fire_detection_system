#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: toggle_vechile_camera.py
#
#   @Author: Shun Li
#
#   @Date: 2021-09-24
#
#   @Email: 2015097272@qq.com
#
#   @Description: open the DJI flight camera image, and seperate image
#
# ------------------------------------------------------------------------------

import sys
import rospy
from dji_osdk_ros.srv import SetupCameraStream
from sensor_msgs.msg import Image
from seperate_original_image import OriginalImageSeperator

class GetImageNode(object):
    def __init__(self):
        self.image_frame = Image()
        self.rate = rospy.Rate(5)

        rospy.wait_for_service("setup_camera_stream")
        self.set_camera_cli = rospy.ServiceProxy("setup_camera_stream",
                                                 SetupCameraStream)

    def image_cb(self, msg):
        self.image_frame = msg

    def run(self, toggle):

        set_camera_handle = SetupCameraStream()

        if toggle == "open":
            result = self.set_camera_cli(
                set_camera_handle._request_class.MAIN_CAM, 1)
            rospy.loginfo("start the main camera stream, "+ str(result))

            rospy.loginfo("start seperate the main aligned images into IR and RGB images!")
            seperator = OriginalImageSeperator()
            seperator.run()


        elif toggle == "close":
            result = self.set_camera_cli(
                set_camera_handle._request_class.MAIN_CAM, 0)
            rospy.loginfo("close the camera stream, "+ str(result))

        else:
            rospy.logerr("Wrong cmd!")


if __name__ == '__main__':

    rospy.init_node('toggle_vechile_camera_node', anonymous=True)

    if len(sys.argv) != 2:
        rospy.logerr("Need 1 cmd!")
    else:
        node = GetImageNode()
        node.run(sys.argv[1])

