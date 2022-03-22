/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: PoseCalculator.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-03-17
 *
 *   @Description:
 *
 *******************************************************************************/

#ifndef INCLUDE_MODULES_POSECALCULATOR_POSECALCULATOR_HPP_
#define INCLUDE_MODULES_POSECALCULATOR_POSECALCULATOR_HPP_

#include <ros/ros.h>
#include <ros/package.h>
#include <stereo_camera_vo/common/camera.h>
#include <stereo_camera_vo/common/frame.h>
#include <stereo_camera_vo/module/frontend.h>

#include "modules/StereoCameraOperator/StereoCamOperator.hpp"
#include "tools/PrintControl/PrintCtrlMacro.h"

#include <memory>
#include <csignal>
#include <string>

#include <sophus/se3.hpp>

namespace FFDS {
namespace MODULES {
class PoseCalculator {
 public:
  PoseCalculator();

  void Step();

 private:
  stereo_camera_vo::common::Camera::Ptr camera_left_;
  stereo_camera_vo::common::Camera::Ptr camera_right_;
  stereo_camera_vo::module::Frontend::Ptr frontend_{nullptr};

  std::shared_ptr<FFDS::MODULES::StereoCamOperator> stereo_cam_operator{
      nullptr};

  Sophus::SE3d Twb2Twc(const geometry_msgs::QuaternionStamped att_body_ros,
                       const Eigen::Vector3d trans);
};
}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_POSECALCULATOR_POSECALCULATOR_HPP_
