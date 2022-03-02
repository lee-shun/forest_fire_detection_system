#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: call_segment_smoke.py
#
#   @Author: Shun Li
#
#   @Date: 2021-12-05
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

import rospy
from forest_fire_detection_system.srv import SegmentSmoke, SegmentSmokeRequest

def call_segment():
    log = Log(__name__).getlog()
    rospy.wait_for_service('forest_fire_detection_system/segment_smoke')
    try:
        segment_smoke_client = rospy.ServiceProxy('forest_fire_detection_system/segment_smoke', SegmentSmoke)
        resp1 = segment_smoke_client(SegmentSmokeRequest(15.0))
    except rospy.ServiceException as e:
        log.error("Service call failed: %s"%e)

if __name__=="__main__":
    rospy.init_node('call_segment_smoke_node', anonymous=True)
    call_segment()
