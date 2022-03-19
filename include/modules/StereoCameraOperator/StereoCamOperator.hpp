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
#include <geometry_msgs/QuaternionStamped.h>

// asyn sensor data
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <string>
#include <memory>

// asyn data struct
#include <common/DepthImgWithPoseAtt.hpp>

// handle setero images
#include <dji_ack.hpp>
#include <dji_osdk_ros/stereo_utility/camera_param.hpp>
#include <dji_osdk_ros/stereo_utility/stereo_frame.hpp>

#include <opencv2/core/core.hpp>

namespace FFDS {
namespace MODULES {
class StereoCamOperator {
 public:
  explicit StereoCamOperator(const std::string m300_stereo_config_path);

  void UpdateOnce() { ros::spinOnce(); }

  const sensor_msgs::PointCloud2& GetRosPtCloudOnce() const {
    return ros_pt_cloud_;
  }
  const cv::Mat& GetRectLeftImgOnce() const { return img_rect_left_; }
  const cv::Mat& GetRectRightImgOnce() const { return img_rect_right_; }
  const geometry_msgs::QuaternionStamped& GetAttOnce() const { return att_; }

  // catch ctrl+c to stop the vga subscription...
  static void ShutDownHandler(int sig_num);

 private:
  ros::NodeHandle nh_;
  ros::ServiceClient stereo_vga_subscription_client_;

  // data to be gotten
  cv::Mat img_rect_left_;
  cv::Mat img_rect_right_;
  geometry_msgs::QuaternionStamped att_;
  sensor_msgs::PointCloud2 ros_pt_cloud_;

  M210_STEREO::CameraParam::Ptr camera_left_ptr_;
  M210_STEREO::CameraParam::Ptr camera_right_ptr_;
  M210_STEREO::StereoFrame::Ptr stereo_frame_ptr_;

  message_filters::Subscriber<sensor_msgs::Image> img_left_sub_;
  message_filters::Subscriber<sensor_msgs::Image> img_right_sub_;
  message_filters::Subscriber<geometry_msgs::QuaternionStamped> attitude_sub_;

  std::shared_ptr<message_filters::TimeSynchronizer<
      sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::QuaternionStamped>>
      imgs_att_synchronizer_{nullptr};

  void StereoImgAttPtCloudCallback(
      const sensor_msgs::ImageConstPtr& img_left,
      const sensor_msgs::ImageConstPtr& img_right,
      const geometry_msgs::QuaternionStampedConstPtr& att);
};
}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_STEREOCAMERAOPERATOR_STEREOCAMOPERATOR_HPP_
