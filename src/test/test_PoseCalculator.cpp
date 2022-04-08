/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: test_PoseCalculator.cpp
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

#include <ros/ros.h>
#include <ros/package.h>

#include "modules/PoseCalculator/PoseCalculator.hpp"
#include "modules/StereoCameraOperator/StereoCamOperator.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "test_pose_calculator_node");
  ros::NodeHandle nh;

  const std::string package_path =
      ros::package::getPath("forest_fire_detection_system");

  const std::string m300_stereo_config_path =
      package_path + "/config/m300_front_stereo_param.yaml";

  FFDS::MODULES::PoseCalculator pose_calculator;
  FFDS::MODULES::StereoCamOperator imu_imgs_grabber(m300_stereo_config_path);

  while (ros::ok()) {
    imu_imgs_grabber.UpdateOnce();

    if (imu_imgs_grabber.GetMessageFilterStatus() ==
        FFDS::MODULES::StereoCamOperator::MessageFilterStatus::EMPTY) {
      PRINT_WARN("no valid message filter! continue ...");
      continue;
    }

    cv::Mat left_img = imu_imgs_grabber.GetRectLeftImgOnce();
    cv::Mat right_img = imu_imgs_grabber.GetRectRightImgOnce();

    geometry_msgs::QuaternionStamped att_body_ros =
        imu_imgs_grabber.GetAttOnce();
    Eigen::Quaterniond q;
    q.w() = att_body_ros.quaternion.w;
    q.x() = att_body_ros.quaternion.x;
    q.y() = att_body_ros.quaternion.y;
    q.z() = att_body_ros.quaternion.z;

    Sophus::SE3d Twb_init(q, Eigen::Vector3d::Zero());

    Sophus::SE3d Tcw_init =
        FFDS::MODULES::PoseCalculator::Twb2Twc(Twb_init).inverse();

    Sophus::SE3d Tcw = pose_calculator.Step(left_img, right_img, Tcw_init);

    Sophus::SE3d Twb = FFDS::MODULES::PoseCalculator::Twc2Twb(Tcw.inverse());

    std::cout << "drone pose after: \n" << Twb.matrix() << std::endl;
  }

  return 0;
}
