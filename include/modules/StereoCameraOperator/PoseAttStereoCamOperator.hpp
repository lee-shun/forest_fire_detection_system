/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: PoseAttStereoCamOperator.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-02-14
 *
 *   @Description:
 *
 *******************************************************************************/

#ifndef INCLUDE_MODULES_STEREOCAMERAOPERATOR_POSEATTSTEREOCAMOPERATOR_HPP_
#define INCLUDE_MODULES_STEREOCAMERAOPERATOR_POSEATTSTEREOCAMOPERATOR_HPP_

#include <ros/ros.h>

// stereo camera
#include <dji_osdk_ros/StereoVGASubscription.h>
#include <sensor_msgs/Image.h>

// local and global pose
#include <dji_osdk_ros/SetLocalPosRef.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/NavSatFix.h>

// attitude
#include <geometry_msgs/QuaternionStamped.h>

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
class PoseAttStereoCamOperator {
 public:
  explicit PoseAttStereoCamOperator(const std::string m300_stereo_cam_param);

  const FFDS::COMMON::DepthImgWithPoseAtt& getDepthWithPoseAttOnce() const;

  static cv::Mat Ros2CVImg(const std::string img_name,
                           const sensor_msgs::ImageConstPtr& img,
                           bool show_img = false);
  // handle ctrl+c shutdown stop the vga...
  static void ShutDownHandler(int sig_num);

 private:
  ros::NodeHandle nh_;

  /**
   * service
   * */
  ros::ServiceClient set_local_pos_ref_client_;
  ros::ServiceClient stereo_vga_subscription_client_;

  /**
   * msg
   * */
  message_filters::Subscriber<sensor_msgs::NavSatFix> gps_sub_;
  message_filters::Subscriber<geometry_msgs::PointStamped> local_pos_sub_;
  message_filters::Subscriber<geometry_msgs::QuaternionStamped> attitude_sub_;
  message_filters::Subscriber<sensor_msgs::Image> img_left_sub_;
  message_filters::Subscriber<sensor_msgs::Image> img_right_sub_;

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::NavSatFix, geometry_msgs::PointStamped,
      geometry_msgs::QuaternionStamped, sensor_msgs::Image, sensor_msgs::Image>
      PoseAttStereoSyncPloicy;

  // 1. gps, 2. local pose, 3. rotation, 4. left image, 5. right image
  message_filters::Synchronizer<PoseAttStereoSyncPloicy>* topic_synchronizer_;

  //! Instantiate some relevant objects
  M210_STEREO::CameraParam::Ptr camera_left_ptr_;
  M210_STEREO::CameraParam::Ptr camera_right_ptr_;
  M210_STEREO::StereoFrame::Ptr stereo_frame_ptr_;

  FFDS::COMMON::DepthImgWithPoseAtt depth_pose_att_;

  void SyncDepthPoseAttCallback(
      const sensor_msgs::NavSatFixConstPtr& gps_msg,
      const geometry_msgs::PointStampedConstPtr& local_pos_msg,
      const geometry_msgs::QuaternionStampedConstPtr& att_msg,
      const sensor_msgs::ImageConstPtr& left_img_msg,
      const sensor_msgs::ImageConstPtr& right_img_msg);

  sensor_msgs::PointCloud2 CalPtCloud(
      const sensor_msgs::ImageConstPtr& img_left,
      const sensor_msgs::ImageConstPtr& img_right);
};
}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_STEREOCAMERAOPERATOR_POSEATTSTEREOCAMOPERATOR_HPP_
