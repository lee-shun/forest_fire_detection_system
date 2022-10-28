/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: GrabDataDepthEstimation.hpp
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

#ifndef INCLUDE_APP_FIRE_POINTS_DEPTH_ESTIMATION_GRABDATADEPTHESTIMATION_HPP_
#define INCLUDE_APP_FIRE_POINTS_DEPTH_ESTIMATION_GRABDATADEPTHESTIMATION_HPP_

#include <dji_osdk_ros/common_type.h>
#include <dji_osdk_ros/ObtainControlAuthority.h>
#include <dji_osdk_ros/FlightTaskControl.h>
#include <dji_osdk_ros/SetJoystickMode.h>

#include <vector>
#include <ros/ros.h>

namespace FFDS {
namespace APP {
class GrabDataDepthEstimationManager {
 public:
  GrabDataDepthEstimationManager();
  void run(const float desired_height, const float desired_length);
  void Grab(int folder_number);

 private:
  bool MoveByPosOffset(dji_osdk_ros::FlightTaskControl &task,
                       const dji_osdk_ros::JoystickCommand &offsetDesired,
                       float posThresholdInM, float yawThresholdInDeg);

  std::vector<dji_osdk_ros::JoystickCommand> GenerateOffsetCommands(
      const float y);

  ros::NodeHandle nh_;

  ros::ServiceClient task_control_client;
  ros::ServiceClient set_joystick_mode_client;
  ros::ServiceClient obtain_ctrl_authority_client;

  dji_osdk_ros::FlightTaskControl control_task;
  dji_osdk_ros::ObtainControlAuthority obtainCtrlAuthority;

  std::atomic<bool> grab_run_;
};
}  // namespace APP
}  // namespace FFDS

#endif  // INCLUDE_APP_FIRE_POINTS_DEPTH_ESTIMATION_GRABDATADEPTHESTIMATION_HPP_
