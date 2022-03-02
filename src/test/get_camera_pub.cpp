/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVLab. All rights reserved.
 *
 *   @Filename: get_camera_pub.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-09-24
 *
 *   @Description:
 *
 ******************************************************************************/
#include <dji_osdk_ros/SetupCameraStream.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <dji_camera_image.hpp>
#include <iostream>

int main(int argc, char** argv) {
  ros::init(argc, argv, "camera_stream_node");
  ros::NodeHandle nh;

  /*! RGB flow init */
  auto setup_camera_stream_client =
      nh.serviceClient<dji_osdk_ros::SetupCameraStream>("setup_camera_stream");
  dji_osdk_ros::SetupCameraStream setupCameraStream_;

  /* start */
  setupCameraStream_.request.cameraType = setupCameraStream_.request.FPV_CAM;
  setupCameraStream_.request.start = 0;
  setup_camera_stream_client.call(setupCameraStream_);

  ros::Duration(20.0).sleep();

  /* end */
  setupCameraStream_.request.cameraType = setupCameraStream_.request.FPV_CAM;
  setupCameraStream_.request.start = 0;
  setup_camera_stream_client.call(setupCameraStream_);

  return 0;
}
