/*******************************************************************************
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: AlianRGB_IR.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-11-23
 *
 *   @Description:
 *
 *******************************************************************************/
#include <cv_bridge/cv_bridge.h>
#include <ros/init.h>
#include <ros/ros.h>
#include <tools/PrintControl/PrintCtrlMacro.h>

#include <opencv2/highgui/highgui.hpp>

/**
 * NOTE: the images here are not captured by the message filter, but since I use
 * NOTE: it in the static scenario*/

cv_bridge::CvImagePtr rgbImgPtr;
cv_bridge::CvImagePtr irImgPtr;
cv::Mat rgbImg;
cv::Mat irImg;

void rgbImgCallback(const sensor_msgs::ImageConstPtr msg) {
  rgbImgPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  rgbImg = rgbImgPtr->image;
}

void irImgCallback(const sensor_msgs::ImageConstPtr msg) {
  irImgPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  irImg = irImgPtr->image;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "alian_rgb_ir_node");
  ros::NodeHandle nh;

  ros::Subscriber rgbSub =
      nh.subscribe("forest_fire_detection_system/main_camera_rgb_image", 10,
                   &rgbImgCallback);
  ros::Subscriber irSub = nh.subscribe(
      "forest_fire_detection_system/main_camera_ir_image", 10, &irImgCallback);
  ros::Duration(3.0).sleep();

  std::string distance_as_name;
  PRINT_INFO("input the distance as the name of the pictures:\n");
  std::cin >> distance_as_name;

  while (ros::ok()) {
    ros::spinOnce();

    PRINT_INFO("the size of rgb image is %dx%d", rgbImg.rows, rgbImg.cols);
    PRINT_INFO("the size of rgb image is %dx%d", irImg.rows, irImg.cols);
    if (rgbImg.empty() || irImg.empty()) {
      PRINT_WARN("wait for image!");
    } else {
      cv::imshow("RGB", rgbImg);
      cv::imshow("IR", irImg);
      cv::waitKey(0);

      cv::imwrite("./RGB_" + distance_as_name + ".jpg", rgbImg);
      cv::imwrite("./IR_" + distance_as_name + ".jpg", irImg);
      PRINT_INFO("images saved...");
      break;
    }

    ros::Rate(1.0).sleep();
  }

  return 0;
}
