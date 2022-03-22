/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: PoseAttStereoCamOperator.cpp
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

#include <modules/StereoCameraOperator/PoseAttStereoCamOperator.hpp>
#include <tools/PrintControl/PrintCtrlMacro.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <opencv2/viz/widgets.hpp>

FFDS::MODULES::PoseAttStereoCamOperator::PoseAttStereoCamOperator(
    const std::string m300_stereo_cam_param) {
  /**
   * Step: 1 init stereo camera handler
   * */
  M210_STEREO::Config::setParamFile(m300_stereo_cam_param);
  camera_left_ptr_ = M210_STEREO::CameraParam::createCameraParam(
      M210_STEREO::CameraParam::FRONT_LEFT);
  camera_right_ptr_ = M210_STEREO::CameraParam::createCameraParam(
      M210_STEREO::CameraParam::FRONT_RIGHT);
  stereo_frame_ptr_ = M210_STEREO::StereoFrame::createStereoFrame(
      camera_left_ptr_, camera_right_ptr_);

  /**
   * Step: 2 services for stereo camera
   * */
  set_local_pos_ref_client_ = nh_.serviceClient<dji_osdk_ros::SetLocalPosRef>(
      "/set_local_pos_reference");
  stereo_vga_subscription_client_ =
      nh_.serviceClient<dji_osdk_ros::StereoVGASubscription>(
          "stereo_vga_subscription");

  PRINT_INFO("Wait for the local ref and stereo_vga to open...");
  set_local_pos_ref_client_.waitForExistence();
  stereo_vga_subscription_client_.waitForExistence();

  /**
   * Step: 3 init stereo camera and pose
   * */
  dji_osdk_ros::SetLocalPosRef set_local_pos_reference;
  set_local_pos_ref_client_.call(set_local_pos_reference);
  if (set_local_pos_reference.response.result) {
    PRINT_INFO("Set local position reference successfully!");
  } else {
    PRINT_ERROR("Set local position reference failed!");
    return;
  }

  dji_osdk_ros::StereoVGASubscription set_stereo_vga_subscription;

  set_stereo_vga_subscription.request.vga_freq =
      set_stereo_vga_subscription.request.VGA_20_HZ;
  set_stereo_vga_subscription.request.front_vga = 1;
  set_stereo_vga_subscription.request.unsubscribe_vga = 0;

  stereo_vga_subscription_client_.call(set_stereo_vga_subscription);
  if (set_stereo_vga_subscription.response.result) {
    PRINT_INFO("Set stereo vga subscription successfully!");
  } else {
    PRINT_ERROR("Set stereo vga subscription failed!");
    return;
  }

  /**
   * Step: 4 use filter to sync the sensors data
   * */
  gps_sub_.subscribe(nh_, "dji_osdk_ros/gps_position", 1);
  local_pos_sub_.subscribe(nh_, "dji_osdk_ros/local_position", 1);
  attitude_sub_.subscribe(nh_, "dji_osdk_ros/attitude", 1);
  img_left_sub_.subscribe(nh_, "dji_osdk_ros/stereo_vga_front_left_images", 1);
  img_right_sub_.subscribe(nh_, "dji_osdk_ros/stereo_vga_front_right_images",
                           1);

  // synchronizer
  topic_synchronizer_ =
      new message_filters::Synchronizer<PoseAttStereoSyncPloicy>(
          PoseAttStereoSyncPloicy(10), gps_sub_, local_pos_sub_, attitude_sub_,
          img_left_sub_, img_right_sub_);

  topic_synchronizer_->registerCallback(
      boost::bind(&PoseAttStereoCamOperator::SyncDepthPoseAttCallback, this, _1,
                  _2, _3, _4, _5));

  ros::Duration(1.0).sleep();
  PRINT_INFO("Create PoseAttStereoCamOperator done!");
}

void FFDS::MODULES::PoseAttStereoCamOperator::SyncDepthPoseAttCallback(
    const sensor_msgs::NavSatFixConstPtr& gps_msg,
    const geometry_msgs::PointStampedConstPtr& local_pos_msg,
    const geometry_msgs::QuaternionStampedConstPtr& att_msg,
    const sensor_msgs::ImageConstPtr& left_img_msg,
    const sensor_msgs::ImageConstPtr& right_img_msg) {
  depth_pose_att_.gps[0] = gps_msg->longitude;
  depth_pose_att_.gps[1] = gps_msg->latitude;
  depth_pose_att_.gps[2] = gps_msg->altitude;

  depth_pose_att_.translation[0] = local_pos_msg->point.x;
  depth_pose_att_.translation[1] = local_pos_msg->point.y;
  depth_pose_att_.translation[2] = local_pos_msg->point.z;

  depth_pose_att_.rotation.w() = att_msg->quaternion.w;
  depth_pose_att_.rotation.x() = att_msg->quaternion.x;
  depth_pose_att_.rotation.y() = att_msg->quaternion.y;
  depth_pose_att_.rotation.z() = att_msg->quaternion.z;

  depth_pose_att_.pt_cloud = CalPtCloud(left_img_msg, right_img_msg);
  depth_pose_att_.pt_cloud.header.frame_id = "front_stereo_camera";
}

sensor_msgs::PointCloud2 FFDS::MODULES::PoseAttStereoCamOperator::CalPtCloud(
    const sensor_msgs::ImageConstPtr& img_left,
    const sensor_msgs::ImageConstPtr& img_right) {
  DJI::OSDK::ACK::StereoVGAImgData img_VGA_img;

  memcpy(&img_VGA_img.img_vec[0], &img_left->data[0],
         sizeof(char) * M210_STEREO::VGA_HEIGHT * M210_STEREO::VGA_WIDTH);
  memcpy(&img_VGA_img.img_vec[1], &img_right->data[0],
         sizeof(char) * M210_STEREO::VGA_HEIGHT * M210_STEREO::VGA_WIDTH);
  img_VGA_img.frame_index = img_left->header.seq;
  img_VGA_img.time_stamp = img_left->header.stamp.nsec;

  stereo_frame_ptr_->readStereoImgs(img_VGA_img);
  stereo_frame_ptr_->rectifyImgs();
  stereo_frame_ptr_->computeDisparityMap();
  stereo_frame_ptr_->filterDisparityMap();
  stereo_frame_ptr_->unprojectROSPtCloud();

  return stereo_frame_ptr_->getROSPtCloud();
}

const FFDS::COMMON::DepthImgWithPoseAtt&
FFDS::MODULES::PoseAttStereoCamOperator::getDepthWithPoseAttOnce() const {
  ros::spinOnce();
  return depth_pose_att_;
}

cv::Mat FFDS::MODULES::PoseAttStereoCamOperator::Ros2CVImg(
    const std::string img_name, const sensor_msgs::ImageConstPtr& img,
    bool show_img) {
  cv_bridge::CvImagePtr img_ptr;
  cv::Mat raw_img;
  img_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  raw_img = img_ptr->image;

  if (show_img) {
    cv::imshow(img_name, raw_img);
    cv::waitKey(1);
  }
  return raw_img;
}

void FFDS::MODULES::PoseAttStereoCamOperator::ShutDownHandler(int sig_num) {
  PRINT_INFO("Caught shutdown signal: %d", sig_num);
  ros::NodeHandle nh;

  ros::ServiceClient stereo_vga_subscription_client =
      nh.serviceClient<dji_osdk_ros::StereoVGASubscription>(
          "stereo_vga_subscription");

  dji_osdk_ros::StereoVGASubscription set_stereo_vga_subscription;
  set_stereo_vga_subscription.request.unsubscribe_vga = 1;

  stereo_vga_subscription_client.call(set_stereo_vga_subscription);
  if (set_stereo_vga_subscription.response.result) {
    PRINT_INFO("Unsubscript stereo vga successfully!");
  } else {
    PRINT_ERROR("Unsubscript stereo vga failed!");
  }
  exit(sig_num);
}
