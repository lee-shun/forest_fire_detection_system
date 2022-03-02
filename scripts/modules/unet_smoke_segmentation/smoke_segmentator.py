#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: smoke_segmentator.py
#
#   @Author: Shun Li
#
#   @Date: 2021-12-03
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

import rospy
from sensor_msgs.msg import Image
from forest_fire_detection_system.srv import SegmentSmoke

import cv2
from cv_bridge import CvBridge, CvBridgeError

import torch
from torch2trt import TRTModule
import albumentations as A
from albumentations.pytorch import ToTensorV2
from develop.semantic_segmentation_unet.model import UNET

from tools.tensor2cv2 import tensor_to_cv, draw_mask
from tools.custom_logger import Log

# The parameters to control the final imgae size
RESIZE_WIDTH = 255
RESIZE_HEIGHT = 255


# TODO: use the error to catch the empty cv_bridge
class SmokeSegmentator(object):
    def __init__(self, use_tensorRT=True):

        self.log = Log(self.__class__.__name__).getlog()

        self.convertor = CvBridge()
        self.ros_image = None
        self.cv_image = None

        # ros stuff
        self.image_sub = rospy.Subscriber(
            "forest_fire_detection_system/main_camera_rgb_transport_image",
            Image,
            self.image_cb,
            queue_size=1)

        # detection model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.val_transforms = A.Compose([
            A.Resize(height=RESIZE_WIDTH, width=RESIZE_HEIGHT),
            A.Normalize(),
            ToTensorV2(),
        ])

        if use_tensorRT:
            self.param_path = PKG_PATH + \
                "scripts/develop/semantic_segmentation_unet/ModelParams/final_trt.pth"
            self.detector = TRTModule().to(self.device)
            self.detector.load_state_dict(torch.load(self.param_path))
        else:
            self.param_path = PKG_PATH + \
                "scripts/develop/semantic_segmentation_unet/ModelParams/final_flight_test_2.pth"
            self.detector = UNET().to(self.device)
            self.detector.load_state_dict(torch.load(self.param_path))
            self.detector.eval()
        self.log.info("loading params from: " + self.param_path)
        self.log.info("initialize done! %s", self.__class__.__name__)

    def image_cb(self, msg):
        self.ros_image = msg
        try:
            self.cv_image = self.convertor.imgmsg_to_cv2(
                self.ros_image, 'bgr8')
        except CvBridgeError as err:
            self.log.warning(err)

    def segment(self, req):

        start_time = rospy.get_time()

        # for save the original video
        output_org_video = cv2.VideoWriter('org_video.avi',
                                           cv2.VideoWriter_fourcc(*'DIVX'), 5,
                                           (RESIZE_WIDTH, RESIZE_HEIGHT))
        # for save the masked video
        output_masked_video = cv2.VideoWriter(
            'mask_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5,
            (RESIZE_WIDTH * 2, RESIZE_HEIGHT))

        while not rospy.is_shutdown():

            time_interval = rospy.get_time() - start_time
            if time_interval >= req.timeOut:
                self.log.info("time up!")
                break

            if self.cv_image is None:
                self.log.warning("Waiting for ros image!")
            else:
                # STEP: 0 subscribe the image, covert to cv image.

                # STEP: 1 convert the cv image to tensor.
                augmentations = self.val_transforms(image=self.cv_image)
                img_ = augmentations['image']
                tensor_img = img_.float().unsqueeze(0).to(self.device)

                # STEP: 2 feed tensor to detector
                with torch.no_grad():
                    preds = torch.sigmoid(self.detector(tensor_img))
                    # NOTE: this valuse is from test
                    tensor_mask = (preds > 0.50)

                # STEP: 3 mask to cv image mask
                cv_mask = tensor_to_cv(tensor_mask[0].cpu())

                # STEP: 4 merge the cv_mask and original cv_mask
                cv_org_img = cv2.resize(self.cv_image,
                                        (RESIZE_WIDTH, RESIZE_HEIGHT))

                # save before merge
                output_org_video.write(cv_org_img)

                cv_final_img = draw_mask(cv_org_img, cv_mask)

                # STEP: 5 show the mask
                cv_3_mask = cv2.merge((cv_mask, cv_mask, cv_mask))
                show_img = cv2.hconcat([cv_final_img, cv_3_mask])
                cv2.imshow('cv_mask', show_img)
                cv2.waitKey(3)

                # STEP: 6 save the masked video.
                output_masked_video.write(show_img)

            rospy.Rate(5).sleep()

        # end of the saving video
        output_org_video.release()
        output_masked_video.release()
        self.log.info("end of the saving masked video!")

        return True

    def run(self):
        segment_smoke_server = rospy.Service(
            "forest_fire_detection_system/segment_smoke", SegmentSmoke,
            self.segment)
        self.log.info("ready to segment the smoke!")
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node("smoke_segmentator_node", anonymous=False)
    detector = SmokeSegmentator(use_tensorRT=False)
    detector.run()
