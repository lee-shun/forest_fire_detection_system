#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: fire_smoke_rgb_resnet_classifier.py
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

import os
import sys

PKG_PATH = os.path.expanduser('~/catkin_ws/src/forest_fire_detection_system/')
sys.path.append(PKG_PATH + 'scripts/')

from tools.custom_logger import Log

import rospy
from sensor_msgs.msg import Image
from forest_fire_detection_system.msg import SingleFireRGB

import cv2
from cv_bridge import CvBridge, CvBridgeError

import torch
from torch2trt import TRTModule
import torchvision.transforms as transforms

from develop.classifier.resnet18.resnet18_model import Resnet18

RESIZE_WIDTH = 255
RESIZE_HEIGHT = 255

class FireSmokeRgbResnetClassifier(object):
    def __init__(self, use_tensorRT=True):

        self.log = Log(self.__class__.__name__).getlog()
        self.convertor = CvBridge()
        self.ros_image = None
        self.cv_image = None

        self.single_fire_rgb = SingleFireRGB()

        # classifier model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.labels = ['fire', 'background', 'smoke']
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # classifier model params
        if use_tensorRT:
            self.param_path = PKG_PATH + "scripts/develop/classifier/resnet18/resnet18_trt.pth"

            self.classifier = TRTModule().to(self.device)
            self.classifier.load_state_dict(torch.load(self.param_path))
        else:
            self.param_path = PKG_PATH + "scripts/develop/classifier/resnet18/Param_resnet18C_1e5_e18.pth"
            self.classifier = Resnet18(img_channels=3,
                                       num_classes=3).to(self.device)
            self.checkpoint = torch.load(self.param_path,
                                         map_location=self.device)
            self.classifier.load_state_dict(
                self.checkpoint['model_state_dict'])
            self.classifier.eval()

        self.log.info("loading params from: " + self.param_path)
        self.log.info("initialize done! %s", self.__class__.__name__)

    def image_cb(self, msg):
        self.ros_image = msg
        try:
            self.cv_image = self.convertor.imgmsg_to_cv2(
                self.ros_image, 'bgr8')
        except CvBridgeError as err:
            self.log.warning(err)

    def classify(self, cv_image):

        # STEP: 1 convert the cv image to tensor.
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_ = self.transform(img)
        tensor_img = torch.unsqueeze(img_, 0).to(self.device)

        # STEP: 2 feed tensor to classifier
        with torch.no_grad():
            preds = self.classifier(tensor_img)
            output_label = torch.topk(preds, 1)
            pred_class = self.labels[int(output_label.indices)]

        self.log.info("classify current image to: %s", pred_class)

        return pred_class

    def run(self):
        # ros
        rospy.wait_for_message(
            "forest_fire_detection_system/main_camera_rgb_image", Image)
        self.image_sub = rospy.Subscriber(
            "forest_fire_detection_system/main_camera_rgb_image",
            Image,
            self.image_cb,
            queue_size=1)
        self.fire_pos_pub = rospy.Publisher(
            "forest_fire_detection_system/single_fire_in_rgb_image",
            SingleFireRGB,
            queue_size=10)
        rospy.sleep(2.0)
        self.log.info("sub and pud registing done...")

        # video saver
        res_video_saver = cv2.VideoWriter('resnet_classifier_resuult.avi',
                                          cv2.VideoWriter_fourcc(*'DIVX'), 5,
                                          (720, 480))

        while not rospy.is_shutdown():
            if self.cv_image is None:
                self.log.warning("Waiting for ros image!")
            else:
                # STEP: 1 classify the image
                pred_class = self.classify(self.cv_image)

                # STEP: 2 save the result
                cv2.putText(self.cv_image, f"Pred: {pred_class}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)
                resized = cv2.resize(self.cv_image, (720, 480))
                res_video_saver.write(resized)

                # STEP: 2 publish
                if pred_class == self.labels[0]:
                    self.single_fire_rgb.target_type = self.single_fire_rgb.IS_FIRE
                elif pred_class == self.labels[1]:
                    self.single_fire_rgb.target_type = self.single_fire_rgb.IS_BACKGROUND
                elif pred_class == self.labels[2]:
                    self.single_fire_rgb.target_type = self.single_fire_rgb.IS_SMOKE
                else:
                    self.single_fire_rgb.target_type = self.single_fire_rgb.IS_UNKNOWN

                self.fire_pos_pub.publish(self.single_fire_rgb)

            rospy.Rate(10).sleep()

        res_video_saver.release()
        self.log.info("end of saveing result!")


if __name__ == '__main__':
    rospy.init_node("fire_smoke_rgb_resnet_classifier_node", anonymous=True)
    detector = FireSmokeRgbResnetClassifier(use_tensorRT=False)
    detector.run()
