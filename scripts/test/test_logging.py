#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: test_logging.py
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

if __name__ == "__main__":
    def hello():
        log = Log(__name__, use_file=False).getlog()
        log.debug("sort the file")
        log.info("sort the file")
        log.warning("sort the file")
        log.error("sort the file")

    hello()
