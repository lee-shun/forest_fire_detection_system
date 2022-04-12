/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: StereoCamOpertaor.cpp
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
#include <tools/PrintControl/PrintCtrlMacro.h>
#include <modules/StereoCameraOperator/StereoCamOperator.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>

bool FFDS::MODULES::StereoCamOperator::stereo_camera_is_open = false;

FFDS::MODULES::StereoCamOperator::StereoCamOperator(
    const std::string m300_stereo_config_path) {
  /**
   * Step: 1 init stereo frame ptr
   * */
  M210_STEREO::Config::setParamFile(m300_stereo_config_path);
  camera_left_ptr_ = M210_STEREO::CameraParam::createCameraParam(
      M210_STEREO::CameraParam::FRONT_LEFT);
  camera_right_ptr_ = M210_STEREO::CameraParam::createCameraParam(
      M210_STEREO::CameraParam::FRONT_RIGHT);

  stereo_frame_ptr_ = M210_STEREO::StereoFrame::createStereoFrame(
      camera_left_ptr_, camera_right_ptr_);

  /**
   * Step: 2 open stereo camera for vga images
   */
  if (!stereo_camera_is_open) {
    stereo_vga_subscription_client_ =
        nh_.serviceClient<dji_osdk_ros::StereoVGASubscription>(
            "stereo_vga_subscription");
    PRINT_INFO(
        "wait for the stereo_vga_subscription to open stereo cameras...");
    stereo_vga_subscription_client_.waitForExistence();

    dji_osdk_ros::StereoVGASubscription set_stereo_vga_subscription;
    set_stereo_vga_subscription.request.vga_freq =
        set_stereo_vga_subscription.request.VGA_20_HZ;
    set_stereo_vga_subscription.request.front_vga = 1;
    set_stereo_vga_subscription.request.unsubscribe_vga = 0;

    stereo_vga_subscription_client_.call(set_stereo_vga_subscription);
    if (set_stereo_vga_subscription.response.result) {
      PRINT_INFO("set stereo vga subscription successfully!");
      stereo_camera_is_open = true;
    } else {
      PRINT_ERROR("set stereo vga subscription failed!");
      stereo_camera_is_open = false;
    }
  } else {
    PRINT_INFO("stereo camera is already open!");
  }

  /**
   * Step: 3 subscribe left and right, and bind them.
   * */
  img_left_sub_.subscribe(nh_, "dji_osdk_ros/stereo_vga_front_left_images",
                          100);
  img_right_sub_.subscribe(nh_, "dji_osdk_ros/stereo_vga_front_right_images",
                           100);
  attitude_sub_.subscribe(nh_, "dji_osdk_ros/attitude", 100);

  imgs_att_synchronizer_ =
      std::make_shared<message_filters::Synchronizer<ImgsAttSyncPloicy>>(
          ImgsAttSyncPloicy(100), img_left_sub_, img_right_sub_, attitude_sub_);

  imgs_att_synchronizer_->registerCallback(boost::bind(
      &StereoCamOperator::StereoImgAttPtCloudCallback, this, _1, _2, _3));

  ros::Duration(1.0).sleep();
  PRINT_INFO("Create StereoCamOperator done!");
}

void FFDS::MODULES::StereoCamOperator::StereoImgAttPtCloudCallback(
    const sensor_msgs::ImageConstPtr &img_left,
    const sensor_msgs::ImageConstPtr &img_right,
    const geometry_msgs::QuaternionStampedConstPtr &att) {
  att_ = *att;
  // copy to dji_osdk_ros
  DJI::OSDK::ACK::StereoVGAImgData img_VGA_img;
  memcpy(&img_VGA_img.img_vec[0], &img_left->data[0],
         sizeof(char) * M210_STEREO::VGA_HEIGHT * M210_STEREO::VGA_WIDTH);
  memcpy(&img_VGA_img.img_vec[1], &img_right->data[0],
         sizeof(char) * M210_STEREO::VGA_HEIGHT * M210_STEREO::VGA_WIDTH);

  img_VGA_img.frame_index = img_left->header.seq;
  img_VGA_img.time_stamp = img_left->header.stamp.nsec;

  stereo_frame_ptr_->readStereoImgs(img_VGA_img);
  stereo_frame_ptr_->rectifyImgs();
  img_rect_left_ = stereo_frame_ptr_->getRectLeftImg();
  img_rect_right_ = stereo_frame_ptr_->getRectRightImg();

  if (img_rect_left_.empty() || img_rect_right_.empty()) {
    message_filter_status_ = MessageFilterStatus::EMPTY;
    return;
  }

  if (use_ptcloud_) {
    stereo_frame_ptr_->computeDisparityMap();
    stereo_frame_ptr_->filterDisparityMap();
    stereo_frame_ptr_->unprojectROSPtCloud();

    ros_pt_cloud_ = stereo_frame_ptr_->getROSPtCloud();
    if (use_ptcloud_filter_) {
      ros_pt_cloud_ = FilterRosPtCloud(ros_pt_cloud_);
    }
  }
}

sensor_msgs::PointCloud2 FFDS::MODULES::StereoCamOperator::FilterRosPtCloud(
    sensor_msgs::PointCloud2 &raw_cloud) {
  // Container for original & filtered data
  pcl::PCLPointCloud2 *cloud = new pcl::PCLPointCloud2();
  pcl::PCLPointCloud2ConstPtr cloud_ptr(cloud);

  // Convert to PCL data type
  pcl_conversions::toPCL(raw_cloud, *cloud);

  // STEP: perform statistical filtering
  // pcl::StatisticalOutlierRemoval<pcl::PCLPointCloud2> sor;
  // sor.setInputCloud(cloud_ptr);
  // sor.setMeanK(30);
  // sor.setStddevMulThresh(1.0);
  // pcl::PCLPointCloud2 *cloud_after_sor = new pcl::PCLPointCloud2();
  // pcl::PCLPointCloud2ConstPtr cloud_after_sor_ptr(cloud_after_sor);
  // sor.filter(*cloud_after_sor);

  // STEP: pass trough
  pcl::PassThrough<pcl::PCLPointCloud2> pass;
  pass.setInputCloud(cloud_ptr);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(2, 8);
  pcl::PCLPointCloud2 *cloud_after_pass = new pcl::PCLPointCloud2();
  pcl::PCLPointCloud2ConstPtr cloud_after_pass_ptr(cloud_after_pass);
  pass.filter(*cloud_after_pass);

  // STEP: perform the voxel filtering
  pcl::VoxelGrid<pcl::PCLPointCloud2> vox;
  vox.setInputCloud(cloud_after_pass_ptr);
  vox.setLeafSize(0.5, 0.5, 0.5);
  pcl::PCLPointCloud2 cloud_after_vox;
  vox.filter(cloud_after_vox);

  // Convert to ROS data type
  sensor_msgs::PointCloud2 output;
  pcl_conversions::moveFromPCL(cloud_after_vox, output);

  return output;
}

// handle ctrl+c shutdown stop the vga...
void FFDS::MODULES::StereoCamOperator::ShutDownHandler(int sig_num) {
  PRINT_INFO("Caught shutdown signal: %d", sig_num);

  if (!stereo_camera_is_open) {
    PRINT_INFO("stereo is already closed!");
    exit(sig_num);
  }

  ros::NodeHandle nh;
  ros::ServiceClient stereo_vga_subscription_client =
      nh.serviceClient<dji_osdk_ros::StereoVGASubscription>(
          "stereo_vga_subscription");

  dji_osdk_ros::StereoVGASubscription set_stereo_vga_subscription;
  set_stereo_vga_subscription.request.unsubscribe_vga = 1;

  stereo_vga_subscription_client.call(set_stereo_vga_subscription);
  if (set_stereo_vga_subscription.response.result) {
    PRINT_INFO("Unsubscript stereo vga successfully!");
    stereo_camera_is_open = false;
  } else {
    PRINT_ERROR("Unsubscript stereo vga failed!");
  }

  exit(sig_num);
}
