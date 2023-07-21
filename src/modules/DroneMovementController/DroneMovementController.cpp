/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: DroneMovementController.cpp
 *
 *   @Author: ShunLi
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 04/07/2023
 *
 *   @Description:
 *
 *******************************************************************************/

#include "modules/DroneMovementController/DroneMovementController.hpp"
#include "modules/BasicController/PIDController.hpp"

namespace FFDS {
namespace MODULES {
bool DroneMovementController::ctrlDroneMoveByImgTarget(const int target_x,
                                                       const int target_y,
                                                       const int max_ctrl_times,
                                                       const int threshold) {
  int ctrl_times = 0;
  PIDController x_pid(0.01, 0, 0, false, false),
      y_pid(0.01, 0, 0, false, false);
  while (ctrl_times++ < max_ctrl_times) {
    float err_x = target_x - heatPosPix.img_x;
    float err_y = target_y - heatPosPix.img_y;
    // return if both are good.
    if (err_x <= threshold && err_y <= threshold) return true;

    if (err_x > threshold) {
      x_pid.ctrl(err_x);
      float x_offset = x_pid.getOutput();
      ctrlDroneMOveByOffset(control_task, {x_offset, 0.0, 0.0, 0.0}, 0.8, 1);
    }
    if (err_y > threshold) {
      y_pid.ctrl(err_y);
      float y_offset = y_pid.getOutput();
      ctrlDroneMOveByOffset(control_task, {0.0, y_offset, 0.0, 0.0}, 0.8, 1);
    }

    ros::spinOnce();
  }

  float err_x = target_x - heatPosPix.img_x;
  float err_y = target_y - heatPosPix.img_y;
  // return if both are good.
  if (err_x <= threshold && err_y <= threshold) return true;
  // check the error first
  return false;
}

bool DroneMovementController::ctrlDroneMOveByOffset(
    dji_osdk_ros::FlightTaskControl &task,
    const dji_osdk_ros::JoystickCommand &offsetDesired, float posThresholdInM,
    float yawThresholdInDeg) {
  task.request.task =
      dji_osdk_ros::FlightTaskControl::Request::TASK_POSITION_AND_YAW_CONTROL;
  task.request.joystickCommand.x = offsetDesired.x;
  task.request.joystickCommand.y = offsetDesired.y;
  task.request.joystickCommand.z = offsetDesired.z;
  task.request.joystickCommand.yaw = offsetDesired.yaw;
  task.request.posThresholdInM = posThresholdInM;
  task.request.yawThresholdInDeg = yawThresholdInDeg;

  task_control_client.call(task);
  return task.response.result;
}

}  // namespace MODULES
}  // namespace FFDS
