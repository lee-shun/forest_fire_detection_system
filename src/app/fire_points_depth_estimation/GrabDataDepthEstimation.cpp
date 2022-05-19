/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: GrabDataDepthEstimation.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-05-19
 *
 *   @Description:
 *
 *******************************************************************************/

#include "modules/H20TIMUPoseGrabber/H20TIMUPoseGrabber.hpp"
#include "tools/PrintControl/FileWritter.hpp"
#include "tools/PrintControl/PrintCtrlMacro.h"
#include "tools/SystemLib.hpp"

#include <forest_fire_detection_system/ToggleGrabDataDepthEstimation.h>
#include <ros/ros.h>

#include <opencv2/imgcodecs.hpp>
#include <thread>

void Grab() {
  // STEP: set local reference position
  ros::NodeHandle nh;

  ros::ServiceClient set_local_pos_ref_client_;
  set_local_pos_ref_client_ = nh.serviceClient<dji_osdk_ros::SetLocalPosRef>(
      "/set_local_pos_reference");
  dji_osdk_ros::SetLocalPosRef set_local_pos_reference;
  set_local_pos_ref_client_.call(set_local_pos_reference);
  if (set_local_pos_reference.response.result) {
    PRINT_INFO("Set local position reference successfully!");
  } else {
    PRINT_ERROR("Set local position reference failed!");
    return;
  }

  // STEP: New directorys
  std::string home = std::getenv("HOME");
  std::string save_path = home + "/m300_grabbed_data";
  FFDS::TOOLS::shellRm(save_path);

  FFDS::TOOLS::shellMkdir(save_path);
  FFDS::TOOLS::shellMkdir(save_path + "/ir");
  FFDS::TOOLS::shellMkdir(save_path + "/rgb");

  // STEP: New files
  FFDS::TOOLS::FileWritter gps_writter(save_path + "/gps.csv", 8);
  FFDS::TOOLS::FileWritter att_writter(save_path + "/att.csv", 8);
  FFDS::TOOLS::FileWritter local_pose_writter(save_path + "/local_pose.csv", 8);
  FFDS::TOOLS::FileWritter time_writter(save_path + "/time_stamp.csv", 8);

  gps_writter.new_open();
  gps_writter.write("index", "lon", "lat", "alt");

  att_writter.new_open();
  att_writter.write("index", "w", "x", "y", "z");

  local_pose_writter.new_open();
  local_pose_writter.write("index", "x", "y", "z");

  time_writter.new_open();
  time_writter.write("index", "sec", "nsec");

  FFDS::MODULES::H20TIMUPoseGrabber grabber;

  int index = 0;
  while (ros::ok()) {
    grabber.UpdateOnce();
    ros::Time time = ros::Time::now();
    time_writter.write(index, time.sec, time.nsec);

    cv::Mat ir_img = grabber.GetIRImageOnce();
    cv::Mat rgb_img = grabber.GetRGBImageOnce();
    cv::imwrite(save_path + "/ir/" + std::to_string(index) + ".png", ir_img);
    cv::imwrite(save_path + "/rgb/" + std::to_string(index) + ".png", rgb_img);

    sensor_msgs::NavSatFix gps = grabber.GetGPSPoseOnce();
    gps_writter.write(index, gps.longitude, gps.latitude, gps.altitude);

    geometry_msgs::PointStamped local = grabber.GetLocalPosOnce();
    local_pose_writter.write(index, local.point.x, local.point.y,
                             local.point.z);

    geometry_msgs::QuaternionStamped att = grabber.GetAttOnce();
    att_writter.write(index, att.quaternion.w, att.quaternion.x,
                      att.quaternion.y, att.quaternion.z);

    ros::Rate(10).sleep();
    ++index;
  }
}

bool GrabService(
    forest_fire_detection_system::ToggleGrabDataDepthEstimation::Request& req,
    forest_fire_detection_system::ToggleGrabDataDepthEstimation::Response&
        res) {
  std::thread grab_thread;

  if (req.start) {
    if (grab_thread.joinable()) {
      PRINT_WARN("already in grabbing!");
      res.result = false;
    } else {
      grab_thread = std::thread(Grab);
      PRINT_INFO("start grabbing data for depth estimation!");
      res.result = true;
    }
  } else {
    if (grab_thread.joinable()) {
      grab_thread.join();
      PRINT_INFO("stop grabbing data for depth estimation!");
      res.result = true;
    } else {
      PRINT_WARN("not in grabbing! No need to join!");
      res.result = false;
    }
  }

  return true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "grab_data_depth_estimation_node");
  ros::NodeHandle nh;

  // STEP: provide the record service
  ros::ServiceServer service =
      nh.advertiseService("/grab_data_depth_estimation", GrabService);
  PRINT_INFO("ready for grabbing data for depth estimation!");

  ros::MultiThreadedSpinner spinner(2);
  spinner.spin();
  return 0;
}
