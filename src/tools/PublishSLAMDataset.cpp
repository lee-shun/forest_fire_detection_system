/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: PublishSLAMDataset.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-03-26
 *
 *   @Description:
 *
 *******************************************************************************/

#include <ros/ros.h>
#include <ros/package.h>

#include "stereo_camera_vo/tool/m300_dataset.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "publish_SLAM_dataset");
  ros::NodeHandle nh;

  const std::string m300_dataset_path = "/media/ls/WORK/slam_m300/m300_data_1";

  stereo_camera_vo::tool::M300Dataset m300_dataset(m300_dataset_path);
  m300_dataset.Init();

  while (ros::ok()) {
    m300_dataset.NextFrame();
    ros::Rate(10).sleep();
  }
  return 0;
}
