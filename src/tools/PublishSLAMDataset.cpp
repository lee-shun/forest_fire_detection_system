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

#include "stereo_camera_vo/tool/m300_dataset.h"
#include "stereo_camera_vo/tool/system_lib.h"

#include "tools/CVROSImg.hpp"
#include "modules/PoseCalculator/PoseCalculator.hpp"

#include <ros/ros.h>
#include <ros/package.h>
#include <geometry_msgs/QuaternionStamped.h>
bool GetTwbAtIndex(const std::string pose_path, const int pose_index,
                   Eigen::Quaterniond* att) {
  std::ifstream pose_fin_;
  pose_fin_.open(pose_path);
  if (!pose_fin_) {
    PRINT_ERROR("can not open %s in given path, no such file or directory!",
                pose_path.c_str());
    return false;
  }

  std::string pose_tmp;
  std::vector<double> q_elements;
  stereo_camera_vo::tool::SeekToLine(pose_fin_, pose_index);
  // read each w, x, y, z, everytime
  for (int i = 0; i < 4; ++i) {
    if (!getline(pose_fin_, pose_tmp, ',')) {
      PRINT_WARN("pose reading error! at index %d", pose_index);
      return false;
    }
    // PRINT_DEBUG("read pose-wxyz:%.8f", std::stod(pose_tmp));
    q_elements.push_back(std::stod(pose_tmp));
  }

  *att = Eigen::Quaterniond(q_elements[0], q_elements[1], q_elements[2],
                            q_elements[3]);

  return true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "publish_SLAM_dataset");
  ros::NodeHandle nh;

  const std::string m300_dataset_path = "/media/ls/WORK/slam_m300/m300_data_1";

  stereo_camera_vo::tool::M300Dataset m300_dataset(m300_dataset_path);
  m300_dataset.Init();

  int pose_index = 0;
  while (ros::ok()) {
    stereo_camera_vo::common::Frame::Ptr p_frame = m300_dataset.NextFrame();

    if (nullptr == p_frame) {
      PRINT_INFO("quit!");
      break;
    }

    ros::Time current_time = ros::Time::now();
    sensor_msgs::ImagePtr ros_img_left =
        FFDS::TOOLS::CV2ROSImg(p_frame->left_img_, "mono8");
    ros_img_left->header.frame_id = "vag_left";
    ros_img_left->header.stamp = current_time;

    sensor_msgs::ImagePtr ros_img_right =
        FFDS::TOOLS::CV2ROSImg(p_frame->right_img_, "mono8");
    ros_img_right->header.frame_id = "vag_right";
    ros_img_right->header.stamp = current_time;

    Eigen::Quaterniond att_body;
    if (!GetTwbAtIndex(m300_dataset_path + "/pose.txt", pose_index,
                       &att_body)) {
      PRINT_ERROR("Quit!");
      break;
    }
    geometry_msgs::QuaternionStamped att_body_ros;
    att_body_ros.quaternion.w = att_body.w();
    att_body_ros.quaternion.x = att_body.x();
    att_body_ros.quaternion.y = att_body.y();
    att_body_ros.quaternion.z = att_body.z();
    att_body_ros.header.stamp = current_time;
    att_body_ros.header.frame_id = "body_FLU";
    std::cout << "att:\n" << att_body_ros << std::endl;

    ros::Rate(10).sleep();
    ++pose_index;
  }

  return 0;
}
