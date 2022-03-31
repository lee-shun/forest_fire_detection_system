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

int main(int argc, char** argv) {
  ros::init(argc, argv, "publish_SLAM_dataset");
  ros::NodeHandle nh;

  ros::Publisher left_img_pub = nh.advertise<sensor_msgs::Image>(
      "dji_osdk_ros/stereo_vga_front_left_images", 1);
  ros::Publisher right_img_pub = nh.advertise<sensor_msgs::Image>(
      "dji_osdk_ros/stereo_vga_front_right_images", 1);
  ros::Publisher att_body_pub = nh.advertise<geometry_msgs::QuaternionStamped>(
      "dji_osdk_ros/attitude", 1);

  const std::string m300_dataset_path = "/media/ls/WORK/slam_m300/m300_data_1";

  stereo_camera_vo::tool::M300Dataset m300_dataset(m300_dataset_path);
  m300_dataset.Init();

  int pose_index = 0;
  while (ros::ok()) {
    ros::spinOnce();

    stereo_camera_vo::common::Frame::Ptr new_frame =
        stereo_camera_vo::common::Frame::CreateFrame();
    if (!m300_dataset.NextFrame(new_frame)) {
      PRINT_INFO("end of publish!");
      break;
    }

    ros::Time current_time = ros::Time::now();
    sensor_msgs::ImagePtr ros_img_left =
        FFDS::TOOLS::CV2ROSImg(new_frame->left_img_, "mono8");
    ros_img_left->header.frame_id = "vag_left";
    ros_img_left->header.stamp = current_time;

    sensor_msgs::ImagePtr ros_img_right =
        FFDS::TOOLS::CV2ROSImg(new_frame->right_img_, "mono8");
    ros_img_right->header.frame_id = "vag_right";
    ros_img_right->header.stamp = current_time;

    Eigen::Quaterniond att_body;
    if (!stereo_camera_vo::tool::M300Dataset::GetAttByIndex(
            m300_dataset_path + "/pose.txt", pose_index, &att_body)) {
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

    left_img_pub.publish(ros_img_left);
    right_img_pub.publish(ros_img_right);
    att_body_pub.publish(att_body_ros);

    ros::Rate(10).sleep();
    ++pose_index;
  }

  return 0;
}
