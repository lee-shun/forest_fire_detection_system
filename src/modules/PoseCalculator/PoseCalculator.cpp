/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: PoseCalculator.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-03-18
 *
 *   @Description:
 *
 *******************************************************************************/

#include "modules/PoseCalculator/PoseCalculator.hpp"

FFDS::MODULES::PoseCalculator::PoseCalculator() {
  /**
   * Step: 1 create stereo camera operator
   * */
  const std::string package_path =
      ros::package::getPath("forest_fire_detection_system");

  const std::string m300_stereo_config_path =
      package_path + "/config/m300_front_stereo_param.yaml";
  stereo_cam_operator = std::make_shared<FFDS::MODULES::StereoCamOperator>(
      m300_stereo_config_path);
  PRINT_INFO("get camera params from %s", m300_stereo_config_path.c_str());

  // regist the shutDownHandler
  signal(SIGINT, FFDS::MODULES::StereoCamOperator::ShutDownHandler);

  /**
   * Step: 2 create the camera instances
   * */
  cv::Mat param_rect_left =
      M210_STEREO::Config::get<cv::Mat>("leftRectificationMatrix");
  cv::Mat param_rect_right =
      M210_STEREO::Config::get<cv::Mat>("rightRectificationMatrix");
  cv::Mat param_proj_left =
      M210_STEREO::Config::get<cv::Mat>("leftProjectionMatrix");
  cv::Mat param_proj_right =
      M210_STEREO::Config::get<cv::Mat>("rightProjectionMatrix");

  double fx = param_proj_left.at<double>(0, 0);
  double fy = param_proj_left.at<double>(1, 1);
  double principal_x = param_proj_left.at<double>(0, 2);
  double principal_y = param_proj_left.at<double>(1, 2);
  double baseline_x_fx = -param_proj_right.at<double>(0, 3);
  double baseline = baseline_x_fx / fx;

  camera_left_ = std::make_shared<stereo_camera_vo::common::Camera>(
      fx, fy, principal_x, principal_y, baseline, Sophus::SE3d());

  Eigen::Vector3d t;
  t << baseline, 0.0f, 0.0f;
  camera_right_ = std::make_shared<stereo_camera_vo::common::Camera>(
      fx, fy, principal_x, principal_y, baseline,
      Sophus::SE3d(Sophus::SO3d(), t));

  /**
   * Step: 3 create the frontend instance
   * */
  const std::string stereo_vo_config_path =
      package_path + "/config/stereo_vo_config.yaml";
  frontend_ = std::make_shared<stereo_camera_vo::module::Frontend>(
      camera_left_, camera_right_, true, stereo_vo_config_path);
  PRINT_INFO("get stereo_vo_config params from %s",
             stereo_vo_config_path.c_str());
}

Sophus::SE3d FFDS::MODULES::PoseCalculator::Twb2Twc(
    const geometry_msgs::QuaternionStamped att_body_ros,
    const Eigen::Vector3d trans) {
  Eigen::Quaterniond att_body, rotate_quat_bc;

  att_body.w() = att_body_ros.quaternion.w;
  att_body.x() = att_body_ros.quaternion.x;
  att_body.y() = att_body_ros.quaternion.y;
  att_body.z() = att_body_ros.quaternion.z;

  rotate_quat_bc.w() = 0.5f;
  rotate_quat_bc.x() = -0.5f;
  rotate_quat_bc.y() = 0.5f;
  rotate_quat_bc.z() = -0.5f;

  Sophus::SE3d T_wb(att_body, trans),
      T_bc(rotate_quat_bc, Eigen::Vector3d::Zero());

  return T_wb * T_bc;
}

void FFDS::MODULES::PoseCalculator::Step() {
  stereo_cam_operator->UpdateOnce();
  auto new_frame = stereo_camera_vo::common::Frame::CreateFrame();

  geometry_msgs::QuaternionStamped att_body_ros =
      stereo_cam_operator->GetAttOnce();

  // leave the translation to be calculated by VO
  Sophus::SE3d init_pose = Twb2Twc(att_body_ros, Eigen::Vector3d::Zero());

  new_frame->SetPose(init_pose);
  new_frame->left_img_ = stereo_cam_operator->GetRectLeftImgOnce();
  new_frame->right_img_ = stereo_cam_operator->GetRectRightImgOnce();

  frontend_->AddFrame(new_frame);

  std::cout << "pose after: \n"
            << new_frame->Pose().matrix().inverse() << std::endl;
}
