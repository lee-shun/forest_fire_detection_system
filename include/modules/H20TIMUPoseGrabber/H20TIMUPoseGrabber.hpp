/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: H20TIMUPoseGrabber.hpp
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

#ifndef INCLUDE_MODULES_H20TIMUPOSEGRABBER_H20TIMUPOSEGRABBER_HPP_
#define INCLUDE_MODULES_H20TIMUPOSEGRABBER_H20TIMUPOSEGRABBER_HPP_

#include <ros/ros.h>

// local and global pose
#include <dji_osdk_ros/SetLocalPosRef.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/NavSatFix.h>

// attitude
#include <geometry_msgs/QuaternionStamped.h>

// gimbal angle
#include <geometry_msgs/Vector3Stamped.h>

// asyn sensor data
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <string>

// asyn data struct
#include <common/DepthImgWithPoseAtt.hpp>

// handle main camera images
#include <dji_ack.hpp>
#include <cv_bridge/cv_bridge.h>

namespace FFDS {
namespace MODULES {
class H20TIMUPoseGrabber {
 public:
  enum MessageFilterStatus { EMPTY, NORMAL };

  H20TIMUPoseGrabber();

  MessageFilterStatus UpdateOnce() {
    ros::spinOnce();
    if (rgb_img_.empty() || ir_img_.empty()) {
      message_filter_status_ = MessageFilterStatus::EMPTY;
    } else {
      message_filter_status_ = MessageFilterStatus::NORMAL;
    }
    return message_filter_status_;
  }

  cv::Mat GetRGBImageOnce() const { return rgb_img_.clone(); }
  cv::Mat GetIRImageOnce() const { return ir_img_.clone(); }

  sensor_msgs::NavSatFix GetGPSPoseOnce() { return gps_pos_; }
  geometry_msgs::PointStamped GetLocalPosOnce() { return local_pos_; }
  geometry_msgs::QuaternionStamped GetAttOnce() { return att_; }
  geometry_msgs::Vector3Stamped GetGimbalOnce() { return gimbal_angle_; }

 private:
  ros::NodeHandle nh_;

  cv_bridge::CvImagePtr rgb_cvimg_ptr_;
  cv_bridge::CvImagePtr ir_cvimg_ptr_;
  cv::Mat rgb_img_;
  cv::Mat ir_img_;

  sensor_msgs::NavSatFix gps_pos_;
  geometry_msgs::PointStamped local_pos_;
  geometry_msgs::QuaternionStamped att_;
  geometry_msgs::Vector3Stamped gimbal_angle_;

  MessageFilterStatus message_filter_status_{MessageFilterStatus::EMPTY};

  /**
   * msg
   * */
  message_filters::Subscriber<sensor_msgs::NavSatFix> gps_sub_;
  message_filters::Subscriber<geometry_msgs::PointStamped> local_pos_sub_;
  message_filters::Subscriber<geometry_msgs::QuaternionStamped> attitude_sub_;
  message_filters::Subscriber<geometry_msgs::Vector3Stamped> gimbal_angle_sub_;
  message_filters::Subscriber<sensor_msgs::Image> img_rgb_sub_;
  message_filters::Subscriber<sensor_msgs::Image> img_ir_sub_;

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::NavSatFix, geometry_msgs::PointStamped,
      geometry_msgs::QuaternionStamped, geometry_msgs::Vector3Stamped,
      sensor_msgs::Image, sensor_msgs::Image>
      PoseAttStereoSyncPloicy;

  // 1. gps, 2. local pose, 3. altitude, 4. gimbal angle 5. rgb image, 6. ir
  // image
  message_filters::Synchronizer<PoseAttStereoSyncPloicy>* topic_synchronizer_;

  void SyncCallback(const sensor_msgs::NavSatFixConstPtr& gps_msg,
                    const geometry_msgs::PointStampedConstPtr& local_pos_msg,
                    const geometry_msgs::QuaternionStampedConstPtr& att_msg,
                    const geometry_msgs::Vector3StampedConstPtr& gimbal_angle_msg,
                    const sensor_msgs::ImageConstPtr& rgb_img_msg,
                    const sensor_msgs::ImageConstPtr& ir_img_msg);
};
}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_H20TIMUPOSEGRABBER_H20TIMUPOSEGRABBER_HPP_
