/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: CVROSImg.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-03-30
 *
 *   @Description:
 *
 *******************************************************************************/

#ifndef INCLUDE_TOOLS_CVROSIMG_HPP_
#define INCLUDE_TOOLS_CVROSIMG_HPP_

#include <tools/PrintControl/PrintCtrlMacro.h>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

#include <string>

namespace FFDS {
namespace TOOLS {

inline cv::Mat& Ros2CVImg(sensor_msgs::ImageConstPtr pROSImg,
                          const std::string image_encodings) {
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(pROSImg, image_encodings);
  } catch (cv_bridge::Exception& e) {
    PRINT_ERROR("cv_bridge exception: %s", e.what());
  }
  return cv_ptr->image;
}

inline sensor_msgs::ImagePtr CV2ROSImg(const cv::Mat& pCVImg,
                                       const std::string image_encodings) {
  return cv_bridge::CvImage(std_msgs::Header(), image_encodings, pCVImg)
      .toImageMsg();
}

}  // namespace TOOLS
}  // namespace FFDS

#endif  // INCLUDE_TOOLS_CVROSIMG_HPP_
