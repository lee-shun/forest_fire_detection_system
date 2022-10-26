/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: H20TIMUPoseGrabber.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-05-18
 *
 *   @Description:
 *
 *******************************************************************************/

#include "modules/H20TIMUPoseGrabber/H20TIMUPoseGrabber.hpp"
#include "tools/PrintControl/PrintCtrlMacro.h"

FFDS::MODULES::H20TIMUPoseGrabber::H20TIMUPoseGrabber() {
  /**
   * Step: use filter to sync the sensors data
   * */
  gps_sub_.subscribe(nh_, "dji_osdk_ros/gps_position", 1);
  local_pos_sub_.subscribe(nh_, "dji_osdk_ros/local_position", 1);
  attitude_sub_.subscribe(nh_, "dji_osdk_ros/attitude", 1);
  gimbal_angle_sub_.subscribe(nh_, "dji_osdk_ros/gimbal_angle", 1);
  img_rgb_sub_.subscribe(
      nh_, "forest_fire_detection_system/main_camera_rgb_image", 1);
  img_ir_sub_.subscribe(nh_,
                        "forest_fire_detection_system/main_camera_ir_image", 1);

  // synchronizer
  topic_synchronizer_ =
      new message_filters::Synchronizer<PoseAttStereoSyncPloicy>(
          PoseAttStereoSyncPloicy(10), gps_sub_, local_pos_sub_, attitude_sub_,
          gimbal_angle_sub_, img_rgb_sub_, img_ir_sub_);

  topic_synchronizer_->registerCallback(boost::bind(
      &H20TIMUPoseGrabber::SyncCallback, this, _1, _2, _3, _4, _5, _6));

  ros::Duration(1.0).sleep();
  PRINT_INFO("Create H20TIMUPoseGrabber done!");
}

void FFDS::MODULES::H20TIMUPoseGrabber::SyncCallback(
    const sensor_msgs::NavSatFixConstPtr& gps_msg,
    const geometry_msgs::PointStampedConstPtr& local_pos_msg,
    const geometry_msgs::QuaternionStampedConstPtr& att_msg,
    const geometry_msgs::Vector3StampedConstPtr& gimbal_angle_msg,
    const sensor_msgs::ImageConstPtr& rgb_img_msg,
    const sensor_msgs::ImageConstPtr& ir_img_msg) {
  att_ = *att_msg;
  gimbal_angle_ = *gimbal_angle_msg;
  gps_pos_ = *gps_msg;
  local_pos_ = *local_pos_msg;

  rgb_cvimg_ptr_ =
      cv_bridge::toCvCopy(rgb_img_msg, sensor_msgs::image_encodings::BGR8);
  rgb_img_ = rgb_cvimg_ptr_->image;

  ir_cvimg_ptr_ =
      cv_bridge::toCvCopy(ir_img_msg, sensor_msgs::image_encodings::BGR8);
  ir_img_ = ir_cvimg_ptr_->image;
}
