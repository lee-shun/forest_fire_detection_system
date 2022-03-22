/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: test_StereoCamOpertaor.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-02-19
 *
 *   @Description:
 *
 *******************************************************************************/

#include <ros/package.h>
#include <tools/PrintControl/PrintCtrlMacro.h>

#include <csignal>
#include <dji_osdk_ros/stereo_utility/point_cloud_viewer.hpp>
#include <modules/StereoCameraOperator/StereoCamOperator.hpp>
#include <common/DepthImgWithPoseAtt.hpp>

#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv) {
  ros::init(argc, argv, "test_stereo_cam_operator_node");
  ros::NodeHandle nh;

  const std::string package_path =
      ros::package::getPath("forest_fire_detection_system");
  const std::string m300_stereo_config_path =
      package_path + "/config/m300_front_stereo_param.yaml";
  PRINT_INFO("get camera params from %s", m300_stereo_config_path.c_str());

  FFDS::MODULES::StereoCamOperator stereo_cam_operator(m300_stereo_config_path);

  // regist the shutDownHandler
  signal(SIGINT, FFDS::MODULES::StereoCamOperator::ShutDownHandler);

  ros::Publisher pt_pub =
      nh.advertise<sensor_msgs::PointCloud2>("/point_cloud/output", 10);
  sensor_msgs::PointCloud2 pt_cloud;

  while (ros::ok()) {
    stereo_cam_operator.UpdateOnce();

    cv::Mat img_left = stereo_cam_operator.GetRectLeftImgOnce();
    cv::Mat img_right = stereo_cam_operator.GetRectRightImgOnce();
    if (!img_left.empty() && !img_left.empty()) {
      cv::imshow("left_rect", img_left);
      cv::imshow("right_rect", img_right);
      cv::waitKey(1);
    }

    geometry_msgs::QuaternionStamped att = stereo_cam_operator.GetAttOnce();
    std::cout << " att.quaternion = \n" << att.quaternion << std::endl;

    pt_cloud = stereo_cam_operator.GetRosPtCloudOnce();
    pt_pub.publish(pt_cloud);
    PRINT_ENTRY("publish!");

    ros::Rate(5).sleep();
  }

  return 0;
}
