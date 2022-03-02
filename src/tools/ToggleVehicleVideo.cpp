/*******************************************************************************
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: ToggleVehicleVideo.cpp
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
#include <dji_osdk_ros/SetupCameraStream.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <iostream>
#include <modules/ImgVideoOperator/RGB_IRSeperator.hpp>

int main(int argc, char** argv) {
  ros::init(argc, argv, "toggle_vehicle_video_node");
  ros::NodeHandle nh;
  auto setup_camera_stream_client =
      nh.serviceClient<dji_osdk_ros::SetupCameraStream>("setup_camera_stream");
  dji_osdk_ros::SetupCameraStream setupCameraStream_;

  if (argc != 2) {
    PRINT_ERROR("Wrong Params number!");
    return 1;
  }

  if (static_cast<std::string>(argv[1]) == "open") {
    setupCameraStream_.request.cameraType = setupCameraStream_.request.MAIN_CAM;
    setupCameraStream_.request.start = 1;
    setup_camera_stream_client.call(setupCameraStream_);
    PRINT_INFO("open the main camera: %d", setupCameraStream_.response.result);

    if (!setupCameraStream_.response.result) {
      PRINT_ERROR("Open vehicle camera stream failed!");
      return 1;
    }

    PRINT_INFO("start separate the ir and RGB image...");

    FFDS::MODULES::RGB_IRSeperator seperator;
    seperator.run();
    return 0;

  } else if (static_cast<std::string>(argv[1]) == "close") {
    setupCameraStream_.request.cameraType = setupCameraStream_.request.MAIN_CAM;
    setupCameraStream_.request.start = 0;
    setup_camera_stream_client.call(setupCameraStream_);
    PRINT_INFO("close the main camera: %d", setupCameraStream_.response.result);
    return 0;
  } else {
    PRINT_ERROR("Wrong Input! usage: open or close!");
    return 1;
  }
}
