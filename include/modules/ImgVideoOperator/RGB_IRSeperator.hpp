/*******************************************************************************
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: RGB_IRSeperator.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-11-27
 *
 *   @Description:
 *
 *******************************************************************************/

#ifndef INCLUDE_MODULES_IMGVIDEOOPERATOR_RGB_IRSEPERATOR_HPP_
#define INCLUDE_MODULES_IMGVIDEOOPERATOR_RGB_IRSEPERATOR_HPP_

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <tools/PrintControl/PrintCtrlMacro.h>

#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <tools/SystemLib.hpp>

namespace FFDS {
namespace MODULES {
class RGB_IRSeperator {
 public:
  RGB_IRSeperator();

  void run();

 private:
  ros::NodeHandle nh;
  image_transport::ImageTransport it{nh};

  ros::Subscriber imgSub;

  image_transport::Publisher imgIRPub;
  image_transport::Publisher imgRGBPub;
  image_transport::Publisher resizeImgRGBPub;

  cv_bridge::CvImagePtr rawImgPtr;
  cv::Mat rawImg;

  int resRGBWid{255};
  int resRGBHet{255};

  void imageCallback(const sensor_msgs::Image::ConstPtr& img);
};
}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_IMGVIDEOOPERATOR_RGB_IRSEPERATOR_HPP_
