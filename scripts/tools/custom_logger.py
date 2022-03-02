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
#------------------------------------------------------------------------------
# @author: mengting gu
# @contact: 1065504814@qq.com
# @time: 2020/11/5 下午6:03
# @file: log.py
# @desc:
#------------------------------------------------------------------------------

import logging
import time
import os
from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("[WARN]", "yellow", attrs=["bold"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("[ERR]", "red", attrs=["bold", "underline"])
        elif record.levelno == logging.INFO:
            prefix = colored("[INFO]", "green", attrs=["bold"])
        elif record.levelno == logging.DEBUG:
            prefix = colored("[DEBUG]", "magenta", attrs=["bold"])
        else:
            return log
        return prefix + log


class Log(object):
    def __init__(self,
                 logger=None,
                 color=True,
                 name="pedescount",
                 log_cate='search',
                 abbrev_name=None,
                 use_file=False):

        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        plain_formatter = logging.Formatter(
            "[%(levelname)s] [%(filename)s:%(lineno)s|in %(funcName)s]] %(message)s",
            datefmt="%m/%d %H:%M:%S")
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(filename)s:%(lineno)s|in %(funcName)s] ", "blue") +
                "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        ch.close()

        if use_file:
            self.log_time = time.strftime("%Y_%m_%d")
            file_dir = os.getcwd() + '/log'
            if not os.path.exists(file_dir):
                os.mkdir(file_dir)
            self.log_path = file_dir
            self.log_name = self.log_path + "/" + log_cate + "." + self.log_time + '.log'
            fh = logging.FileHandler(self.log_name, 'a', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            fh.close()

    def getlog(self):
        return self.logger
