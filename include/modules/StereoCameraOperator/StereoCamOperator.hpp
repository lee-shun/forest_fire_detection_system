/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: StereoCamOperator.hpp
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

#ifndef INCLUDE_MODULES_STEREOCAMERAOPERATOR_STEREOCAMOPERATOR_HPP_
#define INCLUDE_MODULES_STEREOCAMERAOPERATOR_STEREOCAMOPERATOR_HPP_

#include <ros/ros.h>

// stereo camera
#include <dji_osdk_ros/StereoVGASubscription.h>
#include <sensor_msgs/Image.h>

// asyn sensor data
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <string>

// asyn data struct
#include <common/DepthImgWithPoseAtt.hpp>

// handle setero images
#include <dji_ack.hpp>
#include <dji_osdk_ros/stereo_utility/camera_param.hpp>
#include <dji_osdk_ros/stereo_utility/stereo_frame.hpp>

namespace FFDS {
namespace MODULES {
class StereoCamOperator {
 public:
  explicit StereoCamOperator(const std::string m300_stereo_config_path);
  const sensor_msgs::PointCloud2 &GetRosPtCloudOnce() const;

  // catch ctrl+c to stop the vga subscription...
  static void ShutDownHandler(int sig_num);

 private:
  ros::NodeHandle nh_;
  ros::ServiceClient stereo_vga_subscription_client_;

  sensor_msgs::PointCloud2 ros_pt_cloud_;

  M210_STEREO::CameraParam::Ptr camera_left_ptr_;
  M210_STEREO::CameraParam::Ptr camera_right_ptr_;
  M210_STEREO::StereoFrame::Ptr stereo_frame_ptr_;

  message_filters::Subscriber<sensor_msgs::Image> img_left_sub_;
  message_filters::Subscriber<sensor_msgs::Image> img_right_sub_;
  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image>
      *topic_synchronizer_;

  void StereoPtCloudCallback(const sensor_msgs::ImageConstPtr &img_left,
                             const sensor_msgs::ImageConstPtr &img_right);
};
}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_STEREOCAMERAOPERATOR_STEREOCAMOPERATOR_HPP_
