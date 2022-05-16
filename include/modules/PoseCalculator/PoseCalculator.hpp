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
 *   @Description: get the relative pose from the very first frame.
 *
 *******************************************************************************/

#ifndef INCLUDE_MODULES_POSECALCULATOR_POSECALCULATOR_HPP_
#define INCLUDE_MODULES_POSECALCULATOR_POSECALCULATOR_HPP_

#include <ros/ros.h>
#include <ros/package.h>
#include <dji_osdk_ros/stereo_utility/config.hpp>
#include <stereo_camera_vo/common/camera.h>
#include <stereo_camera_vo/common/frame.h>
#include <stereo_camera_vo/module/frontend.h>

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

  bool Stop() {
    if (nullptr != frontend_) frontend_->Stop();
    PRINT_INFO("stop pose calculator!");
    return true;
  }

  bool Step(const cv::Mat& left_img, const cv::Mat& right_img,
            Sophus::SE3d* pose_Tcw);

  static Sophus::SE3d Twb2Twc(const Sophus::SE3d& Twb) {
    Eigen::Quaterniond rotate_quat_bc;

    rotate_quat_bc.w() = 0.5f;
    rotate_quat_bc.x() = -0.5f;
    rotate_quat_bc.y() = 0.5f;
    rotate_quat_bc.z() = -0.5f;

    // camera and body coordinate only have a rotation between them...
    Sophus::SE3d Tbc(rotate_quat_bc, Eigen::Vector3d::Zero());

    return Twb * Tbc;
  }

  static Sophus::SE3d Twc2Twb(const Sophus::SE3d& Twc) {
    Eigen::Quaterniond rotate_quat_bc;

    rotate_quat_bc.w() = 0.5f;
    rotate_quat_bc.x() = -0.5f;
    rotate_quat_bc.y() = 0.5f;
    rotate_quat_bc.z() = -0.5f;

    // camera and body coordinate only have a rotation between them...
    Sophus::SE3d Tbc(rotate_quat_bc, Eigen::Vector3d::Zero());
    return Twc * Tbc.inverse();
  }

  static void convert2Eigen(const cv::Mat proj, Eigen::Matrix3d* K,
                            Eigen::Vector3d* t) {
    (*K) << proj.at<double>(0, 0), proj.at<double>(0, 1), proj.at<double>(0, 2),
        proj.at<double>(1, 0), proj.at<double>(1, 1), proj.at<double>(1, 2),
        proj.at<double>(2, 0), proj.at<double>(2, 1), proj.at<double>(2, 2);

    (*t) << proj.at<double>(0, 3), proj.at<double>(1, 3), proj.at<double>(2, 3);

    (*t) = (*K).inverse() * (*t);
  }

 private:
  bool is_first_frame_{true};
  Sophus::SE3d first_frame_pose_Tcw_;

  stereo_camera_vo::common::Camera::Ptr camera_left_;
  stereo_camera_vo::common::Camera::Ptr camera_right_;
  stereo_camera_vo::module::Frontend::Ptr frontend_{nullptr};
};
}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_POSECALCULATOR_POSECALCULATOR_HPP_
