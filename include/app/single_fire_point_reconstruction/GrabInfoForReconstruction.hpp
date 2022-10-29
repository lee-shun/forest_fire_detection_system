/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: GrabInfoForReconstruction.hpp
 *
 *   @Author: ShunLi
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 23/10/2022
 *
 *   @Description:
 *
 *******************************************************************************/
#ifndef INCLUDE_APP_SINGLE_FIRE_POINT_RECONSTRUCTION_GRABINFOFORRECONSTRUCTION_HPP_
#define INCLUDE_APP_SINGLE_FIRE_POINT_RECONSTRUCTION_GRABINFOFORRECONSTRUCTION_HPP_

#include <ros/ros.h>
#include <modules/H20TIMUPoseGrabber/H20TIMUPoseGrabber.hpp>
#include <modules/GimbalCameraOperator/GimbalCameraOperator.hpp>
#include <modules/WayPointOperator/WpV2Operator.hpp>

namespace FFDS {
namespace APP {
class GrabInfoReconstructionManager {
 public:
  GrabInfoReconstructionManager();
  void Run();
  void Grab();

 private:
  // parameters
  sensor_msgs::NavSatFix center_;
  float radius_, height_, velocity_;
  int num_of_wps_;
  float grab_rate_;

  ros::NodeHandle nh_;
  sensor_msgs::NavSatFix home_;

  // save the files
  std::string root_path_;
  std::string save_path_;

  void initWpV2Setting(
      dji_osdk_ros::InitWaypointV2Setting* initWaypointV2SettingPtr);
  void generateWpV2Actions(
      dji_osdk_ros::GenerateWaypointV2Action* generateWaypointV2ActionPtr, int actionNum);
};
}  // namespace APP
}  // namespace FFDS

#endif  // INCLUDE_APP_SINGLE_FIRE_POINT_RECONSTRUCTION_GRABINFOFORRECONSTRUCTION_HPP_
