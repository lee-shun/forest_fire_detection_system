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
#include "stereo_camera_vo/tool/system_lib.h"

FFDS::MODULES::PoseCalculator::PoseCalculator() {
  /**
   * Step: 1 create stereo camera operator
   * */
  const std::string package_path =
      ros::package::getPath("forest_fire_detection_system");

  const std::string m300_stereo_config_path =
      package_path + "/config/m300_front_stereo_param.yaml";

  /**
   * Step: 2 create the camera instances
   * */
  M210_STEREO::Config::setParamFile(m300_stereo_config_path);
  cv::Mat param_proj_left =
      M210_STEREO::Config::get<cv::Mat>("leftProjectionMatrix");
  cv::Mat param_proj_right =
      M210_STEREO::Config::get<cv::Mat>("rightProjectionMatrix");

  Eigen::Matrix3d left_K, right_K;
  Eigen::Vector3d left_t, right_t;
  convert2Eigen(param_proj_left, &left_K, &left_t);
  convert2Eigen(param_proj_right, &right_K, &right_t);

  // left t
  left_t = Eigen::Vector3d::Zero();

  // Create new cameras
  camera_left_ = std::make_shared<stereo_camera_vo::common::Camera>(
      left_K(0, 0), left_K(1, 1), left_K(0, 2), left_K(1, 2), left_t.norm(),
      Sophus::SE3d(Sophus::SO3d(), left_t));
  camera_right_ = std::make_shared<stereo_camera_vo::common::Camera>(
      right_K(0, 0), right_K(1, 1), right_K(0, 2), right_K(1, 2),
      right_t.norm(), Sophus::SE3d(Sophus::SO3d(), right_t));

  /**
   * Step: 3 create the frontend instance
   * */
  const std::string frontend_config_path =
      package_path + "/config/frontend_config.yaml";
  // read vo parameters from config file
  stereo_camera_vo::module::Frontend::Param frontend_param;

  YAML::Node node = YAML::LoadFile(frontend_config_path);
  frontend_param.num_features_ =
      stereo_camera_vo::tool::GetParam<int>(node, "num_features", 200);
  frontend_param.num_features_init_ =
      stereo_camera_vo::tool::GetParam<int>(node, "num_features_init", 100);
  frontend_param.num_features_tracking_ =
      stereo_camera_vo::tool::GetParam<int>(node, "num_features_tracking", 50);
  frontend_param.num_features_tracking_bad_ =
      stereo_camera_vo::tool::GetParam<int>(node, "num_features_tracking_bad",
                                            40);
  frontend_param.num_features_needed_for_keyframe_ =
      stereo_camera_vo::tool::GetParam<int>(
          node, "num_features_needed_for_keyframe", 80);

  // create frontend
  frontend_ = stereo_camera_vo::module::Frontend::Ptr(
      new stereo_camera_vo::module::Frontend(camera_left_, camera_right_,
                                             frontend_param, false));

  PRINT_INFO("get frontend_config params from %s",
             frontend_config_path.c_str());
}

bool FFDS::MODULES::PoseCalculator::Step(const cv::Mat& left_img,
                                         const cv::Mat& right_img,
                                         Sophus::SE3d* pose_Tcw) {
  if (left_img.empty() || right_img.empty()) {
    PRINT_WARN("no valid stereo images right now!");
    return false;
  }

  auto new_frame = stereo_camera_vo::common::Frame::CreateFrame();
  new_frame->left_img_ = left_img;
  new_frame->right_img_ = right_img;

  new_frame->use_init_pose_ = true;
  new_frame->SetPose(*pose_Tcw);
  frontend_->AddFrame(new_frame);

  *pose_Tcw = new_frame->Pose();

  return true;
}
