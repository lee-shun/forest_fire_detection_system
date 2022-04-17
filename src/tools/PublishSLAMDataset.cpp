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
#include <dji_osdk_ros/StereoVGASubscription.h>

bool g_vga_is_open = false;

bool StereoVGASubscriptionCallback(
    dji_osdk_ros::StereoVGASubscription::Request& request,
    dji_osdk_ros::StereoVGASubscription::Response& response) {
  if (request.unsubscribe_vga == 1) {
    response.result = true;
    g_vga_is_open = false;
    ROS_INFO("unsubscribe stereo vga images");
    return true;
  }

  if (request.front_vga == 1) {
    g_vga_is_open = true;
    response.result = true;
    ros::Duration(1).sleep();
  }

  return true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "publish_SLAM_dataset");
  ros::NodeHandle nh;

  ros::ServiceServer subscribe_stereo_vga_server = nh.advertiseService(
      "stereo_vga_subscription", &StereoVGASubscriptionCallback);

  ros::Publisher left_img_pub = nh.advertise<sensor_msgs::Image>(
      "dji_osdk_ros/stereo_vga_front_left_images", 100);
  ros::Publisher right_img_pub = nh.advertise<sensor_msgs::Image>(
      "dji_osdk_ros/stereo_vga_front_right_images", 100);
  ros::Publisher att_body_pub = nh.advertise<geometry_msgs::QuaternionStamped>(
      "dji_osdk_ros/attitude", 100);

  std::string m300_dataset_path = "/media/ls/WORK/slam_m300/m300_data_2";
  if (2 != argc) {
    PRINT_ERROR("add m300 dataset path!");
    return -1;
  } else {
    m300_dataset_path = argv[1];
    PRINT_INFO("publish dataset: %s", m300_dataset_path.c_str());
  }

  stereo_camera_vo::tool::M300Dataset m300_dataset(m300_dataset_path);
  if (!m300_dataset.Init()) return -1;

  int pose_index = 0;
  while (ros::ok()) {
    ros::spinOnce();

    if (!g_vga_is_open) {
      ros::Duration(0.5).sleep();
      PRINT_INFO("wait for opening vga!");
      continue;
    }

    stereo_camera_vo::common::Frame::Ptr new_frame =
        stereo_camera_vo::common::Frame::CreateFrame();
    if (!m300_dataset.NextFrame(new_frame)) {
      PRINT_INFO("no new frame!");
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
      PRINT_ERROR("pose file reaches end!");
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

    ros::Rate(15).sleep();
    ++pose_index;
  }

  return 0;
}
