/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: test_PoseAttStereoCamOperator.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-02-15
 *
 *   @Description:
 *
 *******************************************************************************/

#include <ros/package.h>
#include <tools/PrintControl/PrintCtrlMacro.h>

#include <csignal>
#include <dji_osdk_ros/stereo_utility/point_cloud_viewer.hpp>
#include <modules/StereoCameraOperator/PoseAttStereoCamOperator.hpp>
#include <common/DepthImgWithPoseAtt.hpp>

#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv) {
  ros::init(argc, argv, "test_pose_att_stereo_cam_operator_node");
  ros::NodeHandle nh;

  const std::string package_path =
      ros::package::getPath("forest_fire_detection_system");
  const std::string m300_stereo_config_path =
      package_path + "/config/m300_front_stereo_param.yaml";
  PRINT_INFO("get camera params from %s", m300_stereo_config_path.c_str());

  FFDS::MODULES::PoseAttStereoCamOperator pose_att_stereo_cam_operator(
      m300_stereo_config_path);
  // regist the shutDownHandler
  signal(SIGINT, FFDS::MODULES::PoseAttStereoCamOperator::ShutDownHandler);

  FFDS::COMMON::DepthImgWithPoseAtt depth_pose_att;

  ros::Publisher pt_pub =
      nh.advertise<sensor_msgs::PointCloud2>("/point_cloud/output", 10);

  while (ros::ok()) {
    ros::spinOnce();
    depth_pose_att = pose_att_stereo_cam_operator.getDepthWithPoseAttOnce();
    pt_pub.publish(depth_pose_att.pt_cloud);

    ros::Rate(10).sleep();
  }

  return 0;
}
